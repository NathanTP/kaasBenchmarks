import threading
import collections
import ray
import random
import abc

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
        """Return the next actor to send a request to"""
        pass

    @abc.abstractmethod
    def update(self, *args):
        """Update the policy with any additional metadata from the last runner used"""
        pass


class PolicyRR(Policy):
    """A simple round-robin policy with no affinity"""
    def __init__(self, nRunner, runnerClass):
        self.runnerClass = runnerClass
        self.lock = threading.Lock()
        self.last = 0
        self.actors = []
        for i in range(nRunner):
            newActor = self.runnerClass.remote()
            permanentScope.append(newActor)
            self.actors.append(newActor)

    def getRunner(self, clientID, *args):
        with self.lock:
            self.last = (self.last + 1) % len(self.actors)
            actor = self.actors[self.last]

        return actor

    def update(self, *args):
        pass


class actorStatus():
    RESERVED = 0
    IDLE = 1
    BUSY = 2

    def __init__(self):
        self.state = actorStatus.IDLE
        self.ref = None


class statusList():
    def __init__(self):
        self.statuses = []
        self.nReserved = 0
        self.lock = threading.Lock()
        self.reservedCv = threading.Condition(lock=self.lock)

    def updateState(self, status, newState):
        assert self.lock.locked()
        if status.state == actorStatus.RESERVED:
            self.nReserved -= 1
            status.state = newState
        elif newState == actorStatus.RESERVED:
            self.nReserved += 1
            status.state = newState
        else:
            status.state = newState


def pickActorBalanced(slist, timeout=None):
    """Given a list of actor statuses, return the first idle actor. Statuses
    are either ray references for busy actors, or None for idle. If timeout is
    provided, None may be returned if there are no free actors within
    timeout."""

    while True:
        with slist.reservedCv:
            if len(slist.statuses) == 0:
                return None

            while slist.nReserved == len(slist.statuses):
                slist.reservedCv.wait()

            outstanding = []
            for i, status in enumerate(slist.statuses):
                if status.state == actorStatus.IDLE:
                    # Found an idle worker
                    slist.updateState(status, actorStatus.RESERVED)
                    return i
                else:
                    if status.state == actorStatus.BUSY:
                        outstanding.append(status.ref)

            assert len(outstanding) != 0

        # Block until at least one actor is idle
        done, notReady = ray.wait(outstanding, fetch_local=False, timeout=timeout)

        with slist.reservedCv:
            if len(done) == 0:
                # There aren't any free workers within the timeout.  This could
                # theoretically be stale, but it probably isn't and we'll let
                # the policy decide if it's worth trying again
                return None
            else:
                idleRunner = None
                for ref in done:
                    for i, status in enumerate(slist.statuses):
                        if status.state == actorStatus.IDLE:
                            # Someone may have processed the actor while we
                            # waited on the lock
                            idleRunner = i
                        elif status.ref == ref:
                            assert status.state == actorStatus.BUSY
                            slist.updateState(status, actorStatus.IDLE)
                            status.ref = None
                            idleRunner = i

                if idleRunner is None:
                    # Our done list is stale, try again
                    continue

                slist.updateState(slist.statuses[idleRunner], actorStatus.RESERVED)
                return idleRunner

    return idleRunner


class PolicyBalance(Policy):
    """Routes requests to actors with potentially multiple clients per
    actor. It will attempt to balance load across the actors based on
    estimated outstanding work."""
    def __init__(self, nRunner, runnerClass):
        self.runnerClass = runnerClass

        # List of Ray references representing stats from dead actors
        self.pendingActorStats = []

        self.actors = collections.deque()
        for i in range(nRunner):
            newActor = self.runnerClass.remote()
            permanentScope.append(newActor)
            self.actors.append(newActor)

        # List of futures to the first return value of the runner. We assume
        # that if any returns are ready, then the runner is done. If None, then
        # the runner is idle.
        self.sList = statusList()
        self.sList.statuses = collections.deque([actorStatus() for i in range(nRunner)])

    def scaleUp(self, newActor=None):
        """Add a worker to this policy"""
        with self.sList.reservedCv:
            if newActor is None:
                newActor = self.runnerClass.remote()
                permanentScope.append(newActor)
            self.actors.append(newActor)
            self.sList.statuses.append(actorStatus())

    def scaleDown(self, kill=True):
        """Remove a worker from this policy"""
        with self.sList.reservedCv:
            self.sList.statuses.popleft()
            toEvict = self.actors.popleft()

            if kill:
                self.pendingActorStats.append(toEvict.getStats.remote())
                toEvict.terminate.remote()

        return toEvict

    def getRunner(self, clientID, **kwargs):
        """Returns an actor suitable for running a request and an opaque handle
        that must be passed to update() along with the clientID and
        respFutures"""
        timeout = kwargs.get('timeout', None)
        idx = pickActorBalanced(self.sList, timeout=timeout)
        if idx is None:
            return None, None
        else:
            with self.sList.reservedCv:
                actor = self.actors[idx]
                status = self.sList.statuses[idx]
            return actor, status

    def update(self, clientID, handle, respFutures):
        status = handle
        if isinstance(respFutures, list):
            status.ref = respFutures[0]
        else:
            status.ref = respFutures
        with self.sList.reservedCv:
            self.sList.updateState(handle, actorStatus.BUSY)
            self.sList.reservedCv.notify()

    def getStats(self):
        """Return a map of clientIDs to profCollection. Resets stats."""
        stats = {}

        for actor in self.actors:
            self.pendingActorStats.append(actor.getStats.remote())

        actorStats = ray.get(self.pendingActorStats)
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

        self.lock = threading.Lock()

        if not self.exclusive:
            self.actors = []
            for i in range(nRunner):
                newActor = self.runnerClass.remote()
                permanentScope.append(newActor)
                self.actors.append(newActor)

        # {clientID -> PolicyBalance()}
        self.clientPools = {}

    def _makeRoom(self, clientID):
        while True:
            with self.lock:
                clientPool = self.clientPools[clientID]
                clientLength = len(clientPool.actors)

                if self.nRunners < self.maxRunners:
                    if self.exclusive:
                        clientPool.scaleUp()
                    else:
                        clientPool.scaleUp(newActor=self.actors.pop())
                    self.nRunners += 1
                    # This is guaranteed to return without blocking since there's a new
                    # idle worker
                    return clientPool.getRunner(clientID)

                # Pick a candidate for eviction. This will be the client with the most
                # actors (ties are broken randomly).
                maxLength = 0
                for cID, pool in self.clientPools.items():
                    if len(pool.actors) > maxLength:
                        maxLength = len(pool.actors)

                candidates = []
                for cID, pool in self.clientPools.items():
                    if len(pool.actors) == maxLength:
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

            # Wouldn't be fair to kill anyone (or we just did), just block
            # until something frees up. Warning, this may block.
            runner = clientPool.getRunner(clientID)
            if runner[0] is None:
                # Something went wrong. Probably someone scaled our pool down to
                # zero while we were waiting. Try again.
                continue
            else:
                return runner

    def getRunner(self, clientID, **kwargs):
        with self.lock:
            if clientID in self.clientPools:
                clientPool = self.clientPools[clientID]
            else:
                clientPool = PolicyBalance(0, self.runnerClass)
                self.clientPools[clientID] = clientPool

        runner, handle = clientPool.getRunner(clientID, timeout=0.01)

        if runner is not None:
            return runner, handle
        else:
            runner = self._makeRoom(clientID)
            if runner[0] is None:
                raise RuntimeError("Couldn't find runner")
            return runner

    def update(self, clientID, handle, respFutures):
        with self.lock:
            self.clientPools[clientID].update(clientID, handle, respFutures)

    def getStats(self):
        stats = {}
        for pool in self.clientPools.values():
            poolStats = pool.getStats()
            util.mergePerClientStats(stats, poolStats)

        return stats
