import collections
import ray
import random
import abc
import asyncio
import infbench
import concurrent.futures
import time

import util

# There is a bug in ray where if actors ever go out of scope, any reference
# held elsewhere can break. We hack around that issue by preventing actors from
# ever leaving scope with this global.
permanentScope = []


async def defaultWaiter(refs, timeout=None, return_when=asyncio.ALL_COMPLETED, returnIdxs=False):
    """Wait for ray object refs "refs" and return a list of indexes of completed references"""
    # We have to explictly wrap the futures in a task, otherwise there is no
    # way to correlate the done list with refFutures
    refFutures = [asyncio.wrap_future(ref.future()) for ref in refs]

    done, pending = await asyncio.wait(refFutures, timeout=timeout, return_when=return_when)

    if returnIdxs:
        return [refFutures.index(doneTask) for doneTask in done]
    else:
        return


class Policy(abc.ABC):
    @abc.abstractmethod
    def __init__(self, nRunner, runnerClass):
        pass

    @abc.abstractmethod
    def getRunner(self, clientID, *args):
        """Returns: (runner, handle)

        runner: The next actor to send a request to
        handle: An opaque handle that must be passed to update() after sending
                a request to the runner."""
        pass

    @abc.abstractmethod
    def update(self, *args):
        """Update the policy with any additional metadata from the last runner used"""
        pass


@ray.remote
class Pool():
    def __init__(self, nRunner, policy, runActor):
        """RunnerPool is responsible for launching run requests.
                - nRunner: Maximum number of actors to allocate
                - policy: Scheduling policy (when using actors)
                - runActor: Actor class to use for runners in this pool

        How it works:
            Policies maintain a pool of actors and decide which one should be
            run next based on clientID and the current state of the system. The
            pool is a remote actor that makes a policy asynchronous, allowing
            multiple clients to send requests simultaneously and for policies
            to arbitrate concurrent requests. It does this by spawining a
            function for each request that sits in its own thread (Ray does
            this for multiple requests to actors). Policies are thread-safe and
            designed to only return when its time to run their request. In
            practice, the Pool actor has many idle threads blocked on a call to
            a policy in either a lock or some other blocking call like
            ray.wait.  Whenever one of these policies unblocks, the thread
            forwards the request to the actor and returns a future representing
            that call.
        """
        self.maxRunners = nRunner

        self.asyncioLoop = asyncio.get_running_loop()
        self.threadPool = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        self.waiter = self.waitForRefs
        # self.waiter = defaultWaiter

        if policy == 'static':
            print("WARNING: the static policy is hard-coded and only useful for manual experiments and debugging")
            self.policy = PolicyStatic(nRunner, runActor, waiter=self.waiter)
        elif policy == 'rr':
            self.policy = PolicyRR(nRunner, runActor, waiter=self.waiter)
        elif policy == 'exclusive':
            self.policy = PolicyAffinity(nRunner, runActor, exclusive=True, waiter=self.waiter)
        elif policy == 'affinity':
            self.policy = PolicyAffinity(nRunner, runActor, exclusive=False, waiter=self.waiter)
        elif policy == 'balance':
            self.policy = PolicyBalance(nRunner, runActor, waiter=self.waiter)
        elif policy == 'hedge':
            self.policy = PolicyHedge(nRunner, runActor, waiter=self.waiter)
        else:
            raise ValueError(f"Unrecognized policy: {policy}")

        self.stats = {}

    async def getStats(self):
        stats = {}
        policyStats = await self.policy.getStats()
        util.mergePerClientStats(stats, policyStats)
        util.mergePerClientStats(stats, self.stats)

        # clear existing stats
        self.stats = {}

        return stats

    async def waitForRefs(self, refs, timeout=None, return_when=asyncio.ALL_COMPLETED, returnIdxs=False):
        """Use an asyncio thread pool to use ray.wait instead of asyncio.wait.
        ray.wait is asymptotically better than asyncio.wait, ~100x faster for resnetKaas."""
        if return_when == asyncio.ALL_COMPLETED:
            numReturn = len(refs)
        elif return_when == asyncio.FIRST_COMPLETED:
            numReturn = 1
        else:
            raise ValueError("Unrecognized return_when argument: " + return_when)

        done, pending = await self.asyncioLoop. \
            run_in_executor(self.threadPool,
                            lambda: ray.wait(refs, num_returns=numReturn, fetch_local=False, timeout=timeout))

        if returnIdxs:
            return [refs.index(doneRef) for doneRef in done]
        else:
            return

    async def run(self, funcName, nReturn, clientID, inputRefs, args, kwargs={}):
        """Run a model. Args and kwargs will be passed to the appropriate runner"""
        if clientID not in self.stats:
            self.stats[clientID] = infbench.profCollection()

        with infbench.timer("t_policy_run", self.stats[clientID]):
            with infbench.timer('t_policy_wait_input', self.stats[clientID]):
                # Block until the inputs are ready
                await self.waiter(inputRefs, return_when=asyncio.ALL_COMPLETED, returnIdxs=False)

            # Get a free runner (may block).
            runActor = None
            retries = 0
            with infbench.timer("t_policy_wait_runner", self.stats[clientID]):
                while runActor is None:
                    runActor, handle = await self.policy.getRunner(clientID)

                    # getRunner should usually return a valid runActor, but certain
                    # race conditions may require a retry. Too many retries is probably
                    # a bug.
                    retries += 1
                    if retries == 10:
                        print("WARNING: Policy being starved: Are you sure getRunner should be failing this often?")

            with infbench.timer("t_policy_runner_invoke", self.stats[clientID]):
                respFutures = getattr(runActor, funcName).options(num_returns=nReturn).remote(*args, **kwargs)

                self.policy.update(clientID, handle, respFutures)

            # Wait until the runner is done before returning, this ensures that
            # anyone waiting on our response (e.g. post()) can immediately
            # ray.get the answer without blocking. This still isn't ideal for
            # multi-node deployments since the caller may still block while the
            # data is fetched to the local data store
            start = time.time()
            if nReturn == 1:
                await self.waiter([respFutures])
            else:
                await self.waiter(respFutures)
            tReqRun = (time.time() - start)*1000
            self.stats[clientID]['t_policy_wait_result'].increment(tReqRun)

            if isinstance(self.policy, PolicyHedge):
                self.policy.updateMeta(clientID, {'t_run': tReqRun})

            return respFutures


class PolicyStatic(Policy):
    def __init__(self, nRunner, runnerClass, waiter=None):
        self.actors = [runnerClass.options(max_concurrency=1).remote() for i in range(nRunner)]
        permanentScope.extend(self.actors)

    async def getRunner(self, clientID, *args):
        if clientID[-3] == 48:
            return self.actors[0], None
        elif clientID[-3] == 49:
            return self.actors[1], None
        else:
            raise RuntimeError("Ya done goofed")

    def update(self, *args):
        pass

    async def getStats(self):
        statFutures = [a.getStats.remote() for a in self.actors]
        perActorStats = await asyncio.gather(*statFutures)

        stats = {}
        for actorStat in perActorStats:
            util.mergePerClientStats(stats, actorStat)

        return stats


class PolicyRR(Policy):
    """A simple round-robin policy with no affinity"""
    def __init__(self, nRunner, runnerClass, waiter=None):
        self.runnerClass = runnerClass
        self.last = 0
        self.actors = []
        for i in range(nRunner):
            # We set the max concurrency to 1 because functions aren't
            # necessarily thread safe (and to match other policies)
            newActor = self.runnerClass.options(max_concurrency=1).remote()

            # Ensure actor is fully booted and ready
            ray.wait([newActor.getStats.remote()], fetch_local=False)

            permanentScope.append(newActor)
            self.actors.append(newActor)

    async def getRunner(self, clientID, *args):
        self.last = (self.last + 1) % len(self.actors)
        actor = self.actors[self.last]

        return actor, None

    def update(self, *args):
        pass

    async def getStats(self):
        statFutures = [a.getStats.remote() for a in self.actors]
        perActorStats = await asyncio.gather(*statFutures)

        stats = {}
        for actorStat in perActorStats:
            util.mergePerClientStats(stats, actorStat)

        return stats


class actorStatus():
    IDLE = 1      # Unused
    PENDING = 2      # No longer running anything, pending assignment to a request
    BUSY = 3      # Currently running (ref is non-none and can be awaited)
    DEAD = 4      # Removed from active queue, do not use

    def __init__(self):
        self.state = actorStatus.IDLE
        self.ref = None


class Runner():
    def __init__(self, runActor):
        self.actor = runActor
        self.state = actorStatus.IDLE
        self.ref = None


class clientInfo():
    def __init__(self):
        self.nMiss = 0

        # We use a rolling average with infinite history because its easy and
        # probably fine (the experiments aren't really time-varying)
        self.latSum = 0
        self.nSample = 0
        self.latEstimate = 0

    def expectedWait(self):
        if self.nSample == 0:
            return None
        else:
            return self.latEstimate * 1.5

    def updateWait(self, lat):
        self.latSum += lat
        self.latEstimate = self.latSum / self.nSample

    def updateMiss(self, miss):
        if miss:
            self.nMiss += 1
        else:
            self.nMiss -= 1

    def resetMiss(self):
        self.nMiss = 0

    def shouldScale(self):
        # XXX This is super naive. I should think harder about the policy here
        if self.nMiss > 10:
            return True
        else:
            return False


class PolicyHedge(Policy):
    """Attempts to route requests from clients to the same set of GPUs if
    possible. If a long tail is detected, the system begins to hedge on runners
    from outside its affinity pool."""
    def __init__(self, nRunner, runnerClass, waiter=None):
        self.runnerClass = runnerClass
        self.waiter = waiter

        self.allRunners = []
        for i in range(nRunner):
            # We set the max concurrency to 1 because functions aren't
            # necessarily thread safe (and to match other policies)
            newActor = self.runnerClass.options(max_concurrency=1).remote()

            # Ensure actor is fully booted and ready
            ray.wait([newActor.getStats.remote()], fetch_local=False)

            permanentScope.append(newActor)
            self.allRunners.append(Runner(newActor))
        self.unassigned = self.allRunners.copy()

        # {clientID -> [Runner]}
        self.affinityGroups = {}

        # {clientID -> clientInfo}
        self.clientInfos = {}

    def scaleClient(self, clientID):
        if len(self.unassigned) > 0:
            newRunner = self.unassigned.pop()
            self.affinityGroups[clientID].append(newRunner)
            return

        maxSize = 0
        for affGroup in self.affinityGroups.values():
            if len(affGroup) > maxSize:
                maxSize = len(affGroup)

        # Check if it would be fair to scale someone else down
        if maxSize >= len(self.affinityGroups[clientID]):
            return

        # Only consider clients with the most workers
        candidates = []
        for cID, affGroup in self.affinityGroups.items():
            if len(affGroup) == maxSize:
                candidates.append(cID)

        # If we still have multiple candidates, choose randomly
        lot = random.randrange(0, len(candidates))
        victimID = candidates[lot]
        evictedActor = self.affinityGroups[victimID].pop()
        self.affinityGroups[clientID].append(evictedActor)

    async def getRunner(self, clientID, **kwargs):
        if clientID not in self.affinityGroups:
            self.affinityGroups[clientID] = []
            self.clientInfos[clientID] = clientInfo()

        affGroup = self.affinityGroups[clientID]
        cInfo = self.clientInfos[clientID]

        # Every client deserves at least one runner
        if len(affGroup) == 0:
            self.scaleClient(clientID)

        # First try to find an idle worker in the current affinity group
        for runner in affGroup:
            if runner.state == actorStatus.IDLE:
                runner.state = actorStatus.BUSY
                cInfo.updateMiss(False)
                return runner.actor, runner

        # No obviously free workers, try running in our affinity group
        timeout = cInfo.expectedWait()
        doneIdxs = await self.waiter([runner.ref for runner in affGroup], timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED, returnIdxs=True)

        for idx in doneIdxs:
            affGroup[idx].state = actorStatus.IDLE

        if len(doneIdxs) > 0:
            runner = affGroup[doneIdxs[0]]
            runner.state = actorStatus.BUSY
            cInfo.updateMiss(False)
            return runner.actor, runner

        # Didn't get any workers within a reasonable time frame, try stealing
        # from the whole pool
        cInfo.updateMiss(True)
        if cInfo.shouldScale():
            self.scaleClient(clientID)
            cInfo.resetMiss()

        doneIdxs = await self.waiter([runner.ref for runner in self.allRunners], timeout=None,
                                     return_when=asyncio.FIRST_COMPLETED, returnIdxs=True)

        assert len(doneIdxs) > 0
        runner = self.allRunners[doneIdxs[0]]
        runner.state = actorStatus.BUSY
        return runner.actor, runner

    def update(self, clientID, handle, respFutures):
        runner = handle
        runner.ref = respFutures

    def updateMeta(self, clientID, updates):
        self.clientInfos[clientID].updateWait(updates['t_run'])


class ScalableBalance(Policy):
    """Like PolicyBalance but supports scaling up and down."""
    def __init__(self, nRunner, runnerClass, waiter=None):
        self.runnerClass = runnerClass

        self.waiter = waiter
        assert waiter is not None

        # List of Ray references representing stats from dead actors
        self.pendingActorStats = []

        self.runners = collections.deque()
        for i in range(nRunner):
            newActor = self.runnerClass.remote()

            # Ensure actor is fully booted and ready
            ray.wait([newActor.getStats.remote()], fetch_local=False)

            permanentScope.append(newActor)
            self.runners.append((newActor, actorStatus()))

    def scaleUp(self, newActor=None):
        """Add a worker to this policy"""
        if newActor is None:
            newActor = self.runnerClass.remote()
            permanentScope.append(newActor)
        self.runners.append((newActor, actorStatus()))

    def scaleDown(self, kill=True):
        """Remove a worker from this policy"""
        actor, status = self.runners.popleft()

        status.state = actorStatus.DEAD
        if kill:
            self.pendingActorStats.append(actor.getStats.remote())
            actor.terminate.remote()

        return actor

    async def getRunner(self, clientID, **kwargs):
        """Returns an actor suitable for running a request and an opaque handle
        that must be passed to update() along with the clientID and
        respFutures"""
        timeout = kwargs.get('timeout', None)

        if len(self.runners) == 0:
            return None, None

        outstanding = []
        for actor, status in self.runners:
            if status.state == actorStatus.IDLE:
                status.state = actorStatus.BUSY
                return actor, status
            else:
                outstanding.append((actor, status))
        assert len(outstanding) != 0

        # Block until at least one actor is idle
        outstandingRefs = [runner[1].ref for runner in outstanding]
        doneIdxs = await self.waiter(outstandingRefs, timeout=timeout,
                                     return_when=asyncio.FIRST_COMPLETED, returnIdxs=True)

        if len(doneIdxs) == 0:
            # There aren't any free workers within the timeout.
            return None, None
        else:
            idleActor = None
            idleStatus = None
            for doneIdx in doneIdxs:
                doneActor, doneStatus = outstanding[doneIdx]
                if doneStatus.state != actorStatus.DEAD:
                    actorStatus.state = actorStatus.IDLE
                    actorStatus.ref = None
                    idleActor = doneActor
                    idleStatus = doneStatus

            if idleActor is None:
                # Every actor that was busy when we started waiting for
                # outstanding has since been evicted.
                return None, None

            idleStatus.state = actorStatus.BUSY
            return idleActor, idleStatus

    def update(self, clientID, handle, respFutures):
        status = handle
        if isinstance(respFutures, list):
            status.ref = respFutures[0]
        else:
            status.ref = respFutures

    async def getStats(self):
        """Return a map of clientIDs to profCollection. Resets stats."""
        stats = {}

        for actor, _ in self.runners:
            self.pendingActorStats.append(actor.getStats.remote())

        actorStats = await asyncio.gather(*self.pendingActorStats)
        for actorStat in actorStats:
            util.mergePerClientStats(stats, actorStat)

        self.pendingActorStats = []
        return stats


class PolicyBalance(Policy):
    """Routes requests to actors with potentially multiple clients per
    actor. It will attempt to balance load across the actors based on
    estimated outstanding work."""
    def __init__(self, nRunner, runnerClass, waiter=None):
        self.runnerClass = runnerClass

        self.waiter = waiter
        assert waiter is not None

        # List of Ray references representing stats from dead actors
        self.pendingActorStats = []

        self.readyQueue = asyncio.Queue(maxsize=nRunner)
        self.runners = collections.deque()
        for i in range(nRunner):
            newActor = self.runnerClass.remote()

            # Ensure actor is fully booted and ready
            ray.wait([newActor.getStats.remote()], fetch_local=False)

            newStatus = actorStatus()
            newStatus.state = actorStatus.PENDING
            permanentScope.append(newActor)
            self.runners.append((newActor, newStatus))
            self.readyQueue.put_nowait((newActor, newStatus))

        self.emptyPoolEvent = asyncio.Event()
        self.readyQueueTask = asyncio.create_task(self.waitForRunners())

    async def waitForRunners(self):
        """A persistent task that fills the readyQueue as runners complete"""
        while True:
            if len(self.runners) == 0:
                await self.emptyPoolEvent.wait()

            outstanding = []
            while len(outstanding) == 0:
                for actor, status in self.runners:
                    if status.state == actorStatus.BUSY:
                        outstanding.append((actor, status))

                if len(outstanding) == 0:
                    # All runners are PENDING
                    try:
                        await self.readyQueue.join()
                    except asyncio.CancelledError:
                        return
                    assert self.readyQueue.qsize() == 0
            assert len(outstanding) != 0

            # Block until at least one actor is idle
            outstandingRefs = [runner[1].ref for runner in outstanding]
            try:
                doneIdxs = await self.waiter(outstandingRefs, timeout=None,
                                             return_when=asyncio.FIRST_COMPLETED, returnIdxs=True)
            except asyncio.CancelledError:
                break

            for idx in doneIdxs:
                doneActor, doneStatus = outstanding[idx]
                if doneStatus.state == actorStatus.DEAD:
                    continue

                doneStatus.ref = None
                doneStatus.state = actorStatus.PENDING
                self.readyQueue.put_nowait(outstanding[idx])

    async def getRunner(self, clientID, **kwargs):
        """Returns an actor suitable for running a request and an opaque handle
        that must be passed to update() along with the clientID and
        respFutures"""
        if len(self.runners) == 0:
            return None, None

        try:
            # runner, status = await asyncio.wait_for(self.readyQueue.get(), timeout=timeout)
            runner, status = await self.readyQueue.get()
        except asyncio.TimeoutError:
            return None, None
        self.readyQueue.task_done()

        # It's possible that a runner is killed while waiting in the
        # readyQueue. Hopefully this doesn't happen too often.
        if status.state == actorStatus.DEAD:
            # self.readyQueue.task_done()
            return None, None
        else:
            return runner, status

    def update(self, clientID, handle, respFutures):
        status = handle
        status.state = actorStatus.BUSY
        # self.readyQueue.task_done()
        if isinstance(respFutures, list):
            status.ref = respFutures[0]
        else:
            status.ref = respFutures

    async def getStats(self):
        """Return a map of clientIDs to profCollection. Resets stats."""
        stats = {}

        for actor, _ in self.runners:
            self.pendingActorStats.append(actor.getStats.remote())

        actorStats = await asyncio.gather(*self.pendingActorStats)
        for actorStat in actorStats:
            util.mergePerClientStats(stats, actorStat)

        self.pendingActorStats = []
        return stats


class PolicyAffinity(Policy):
    def __init__(self, nRunner, runnerClass, exclusive=True, waiter=None):
        """This policy provides exclusive access to a pool of actors for each
        client. Requests from two clients will never go to the same actor.
        Pools are sized to maximize fairness. If more clients register than
        available GPUs, the system will kill existing actors to make room."""
        self.waiter = waiter
        assert waiter is not None

        self.maxRunners = nRunner
        self.nRunners = 0
        self.runnerClass = runnerClass
        self.exclusive = exclusive

        if not self.exclusive:
            self.actors = []
            for i in range(self.maxRunners):
                newActor = self.runnerClass.remote()

                # Ensure actor is fully booted and ready
                ray.wait([newActor.getStats.remote()], fetch_local=False)

                permanentScope.append(newActor)
                self.actors.append(newActor)

        # {clientID -> ScalableBalance()}
        self.clientPools = {}

    async def _makeRoom(self, clientID):
        while True:
            clientPool = self.clientPools[clientID]
            clientLength = len(clientPool.runners)

            if self.nRunners < self.maxRunners:
                if self.exclusive:
                    clientPool.scaleUp()
                else:
                    clientPool.scaleUp(newActor=self.actors.pop())
                self.nRunners += 1
                runner = await clientPool.getRunner(clientID)
                if runner[0] is None:
                    continue
                else:
                    return runner

            # Pick a candidate for eviction. This will be the client with the most
            # actors (ties are broken randomly).
            maxLength = 0
            for cID, pool in self.clientPools.items():
                if len(pool.runners) > maxLength:
                    maxLength = len(pool.runners)

            candidates = []
            for cID, pool in self.clientPools.items():
                if len(pool.runners) == maxLength:
                    candidates.append(cID)

            # If the client doesn't have workers, we have to give it at least
            # one. Otherwise, if we can make the system more balanced by
            # stealing from another pool, do that. scaling at clientLength <
            # maxLength can make them trade places (c0 has 1, c2 has 2, they
            # will just constantly alternate, benefiting no one).
            if clientLength == 0 or (clientLength + 1) < maxLength:
                # Gotta be somewhat fair. Real fairness is a problem for
                # another day
                lot = random.randrange(0, len(candidates))
                candidate = candidates[lot]
                victimPool = self.clientPools[candidate]
                evictedActor = victimPool.scaleDown(kill=self.exclusive)

                if self.exclusive:
                    clientPool.scaleUp()
                else:
                    clientPool.scaleUp(newActor=evictedActor)

            runner = await clientPool.getRunner(clientID)
            if runner[0] is None:
                # Something went wrong. Probably someone scaled our pool down to
                # zero while we were waiting. Try again.
                continue
            else:
                return runner

    async def getRunner(self, clientID, **kwargs):
        if clientID in self.clientPools:
            clientPool = self.clientPools[clientID]
        else:
            clientPool = ScalableBalance(0, self.runnerClass, waiter=self.waiter)
            self.clientPools[clientID] = clientPool

        runner, handle = await clientPool.getRunner(clientID, timeout=0.01)

        if runner is not None:
            return runner, handle
        else:
            runner = await self._makeRoom(clientID)
            if runner[0] is None:
                raise RuntimeError("Couldn't find runner")
            return runner

    def update(self, clientID, handle, respFutures):
        self.clientPools[clientID].update(clientID, handle, respFutures)

    async def getStats(self):
        stats = {}
        poolStats = await asyncio.gather(*[pool.getStats() for pool in self.clientPools.values()])
        for poolStat in poolStats:
            util.mergePerClientStats(stats, poolStat)

        return stats
