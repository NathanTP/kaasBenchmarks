import collections
import ray
import random
import abc
import asyncio

import util

# There is a bug in ray where if actors ever go out of scope, any reference
# held elsewhere can break. We hack around that issue by preventing actors from
# ever leaving scope with this global.
permanentScope = []


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

        if policy == 'static':
            print("WARNING: the static policy is hard-coded and only useful for manual experiments and debugging")
            self.policy = PolicyStatic(nRunner, runActor)
        elif policy == 'rr':
            self.policy = PolicyRR(nRunner, runActor)
        elif policy == 'exclusive':
            self.policy = PolicyAffinity(nRunner, runActor, exclusive=True)
        elif policy == 'affinity':
            self.policy = PolicyAffinity(nRunner, runActor, exclusive=False)
        elif policy == 'balance':
            self.policy = PolicyBalance(nRunner, runActor)
        else:
            raise ValueError("Unrecognized policy: " + policy)

    async def getStats(self):
        return await self.policy.getStats()

    async def run(self, funcName, nReturn, clientID, inputRefs, args, kwargs={}):
        """Run a model. Args and kwargs will be passed to the appropriate runner"""
        # Block until the inputs are ready
        await asyncio.wait(inputRefs)

        # Get a free runner (may block).
        runActor = None
        retries = 0
        while runActor is None:
            runActor, handle = await self.policy.getRunner(clientID)

            # getRunner should usually return a valid runActor, but certain
            # race conditions may require a retry. Too many retries is probably
            # a bug.
            retries += 1
            if retries == 10:
                print("WARNING: Policy being starved: Are you sure getRunner should be failing this often?")

        respFutures = getattr(runActor, funcName).options(num_returns=nReturn).remote(*args, **kwargs)

        self.policy.update(clientID, handle, respFutures)

        # Wait until the runner is done before returning, this ensures that
        # anyone waiting on our response (e.g. post()) can immediately
        # ray.get the answer without blocking. This still isn't ideal for
        # multi-node deployments since the caller may still block while the
        # data is fetched to the local data store
        if nReturn == 1:
            await asyncio.wait([respFutures])
        else:
            await asyncio.wait(respFutures)

        return respFutures


class PolicyStatic(Policy):
    def __init__(self, nRunner, runnerClass):
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
    def __init__(self, nRunner, runnerClass):
        self.runnerClass = runnerClass
        self.last = 0
        self.actors = []
        for i in range(nRunner):
            # We set the max concurrency to 1 because functions aren't
            # necessarily thread safe (and to match other policies)
            newActor = self.runnerClass.options(max_concurrency=1).remote()
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
    BUSY = 2      # Currently running (ref is non-none and can be awaited)
    DEAD = 3      # Removed from active queue, do not use

    def __init__(self):
        self.state = actorStatus.IDLE
        self.ref = None


class statusList():
    def __init__(self):
        self.statuses = []


class PolicyBalance(Policy):
    """Routes requests to actors with potentially multiple clients per
    actor. It will attempt to balance load across the actors based on
    estimated outstanding work."""
    def __init__(self, nRunner, runnerClass):
        self.runnerClass = runnerClass

        # List of Ray references representing stats from dead actors
        self.pendingActorStats = []

        self.runners = collections.deque()
        for i in range(nRunner):
            newActor = self.runnerClass.remote()
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
                if status.state == actorStatus.BUSY:
                    outstanding.append((actor, status))
        assert len(outstanding) != 0

        # Block until at least one actor is idle
        outstandingRefs = [asyncio.wrap_future(runner[1].ref.future()) for runner in outstanding]
        done, notReady = await asyncio.wait(outstandingRefs,
                                            timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

        if len(done) == 0:
            # There aren't any free workers within the timeout.
            return None, None
        else:
            idleActor = None
            idleStatus = None
            for doneTask in done:
                doneIdx = outstandingRefs.index(doneTask)
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


class PolicyAffinity(Policy):
    def __init__(self, nRunner, runnerClass, exclusive=True):
        """This policy provides exclusive access to a pool of actors for each
        client. Requests from two clients will never go to the same actor.
        Pools are sized to maximize fairness. If more clients register than
        available GPUs, the system will kill existing actors to make room."""
        self.maxRunners = nRunner
        self.nRunners = 0
        self.runnerClass = runnerClass
        self.exclusive = exclusive

        if not self.exclusive:
            self.actors = []
            for i in range(self.maxRunners):
                newActor = self.runnerClass.remote()
                permanentScope.append(newActor)
                self.actors.append(newActor)

        # {clientID -> PolicyBalance()}
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

            if clientLength < maxLength:
                # Gotta be somewhat fair. Real fairness is a problem for
                # another day
                lot = random.randrange(0, len(candidates))
                candidate = candidates[lot]
                # print(f"EVICTING {candidate} for {clientID} ({lot} from choices {candidates})")
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
            clientPool = PolicyBalance(0, self.runnerClass)
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
