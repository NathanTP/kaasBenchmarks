import numpy as np
import collections
import json
import contextlib
import time


class prof():
    def __init__(self, fromDict=None, detail=True):
        """A profiler object for a metric or event type. The counter can be
        updated multiple times per event, while calling increment() moves on to
        a new event. If detail==true, all events are logged allowing more
        complex statistics. This may affect performance if there are many
        events."""
        self.detail = detail
        if fromDict is not None:
            if self.detail:
                self.events = fromDict['events']
            self.total = fromDict['total']
            self.nevent = fromDict['nevent']
        else:
            self.total = 0.0
            self.nevent = 0

            if self.detail:
                self.currentEvent = 0.0
                self.events = []

    def update(self, n):
        """Update increases the value of this entry for the current event."""
        self.total += n
        if self.detail:
            self.currentEvent += n

    def increment(self, n=0):
        """Finalize the current event (increment the event counter). If n is
        provided, the current event will be updated by n before finalizing."""
        self.update(n)
        self.nevent += 1
        if self.detail:
            self.events.append(self.currentEvent)
            self.currentEvent = 0.0

    def report(self):
        """Report the average value per event"""
        rep = {}
        rep['total'] = self.total
        rep['mean'] = self.total / self.nevent
        if self.detail:
            events = np.array(self.events)
            rep['min'] = events.min()
            rep['max'] = events.max()
            rep['p50'] = np.quantile(events, 0.50)
            rep['p90'] = np.quantile(events, 0.90)
            rep['p99'] = np.quantile(events, 0.99)
            rep['events'] = self.events

        return rep


class profCollection(collections.abc.MutableMapping):
    """This is basically a dictionary and can be used anywhere a dictionary of
    profs was previously used. It has a few nice additional features though. In
    particular, it will generate an empty prof whenever a non-existant key is
    accessed."""

    def __init__(self, detail=True):
        # a map of modules included in these stats. Each module is a
        # profCollection. Submodules can nest indefinitely.
        self.detail = detail

        self.mods = {}

        self.profs = dict()

    def __contains__(self, key):
        return key in self.profs

    def __getitem__(self, key):
        if key not in self.profs:
            self.profs[key] = prof(detail=self.detail)
        return self.profs[key]

    def __setitem__(self, key, value):
        self.profs[key] = value

    def __delitem__(self, key):
        del self.profs[key]

    def __iter__(self):
        return iter(self.profs)

    def __len__(self):
        return len(self.profs)

    def __str__(self):
        return json.dumps(self.report(), indent=4)

    def mod(self, name):
        if name not in self.mods:
            self.mods[name] = profCollection(detail=self.detail)

        return self.mods[name]

    def merge(self, new, prefix=''):
        # Start by merging the direct stats
        for k, v in new.items():
            newKey = prefix+k
            if newKey in self.profs:
                self.profs[newKey].increment(v.total)
            else:
                self.profs[newKey] = v

        # Now recursively handle modules
        for name, mod in new.mods.items():
            # Merging into an empty profCollection causes a deep copy
            if name not in self.mods:
                self.mods[name] = profCollection(detail=self.detail)
            self.mods[name].merge(mod)

    def report(self):
        flattened = {name: v.report() for name, v in self.profs.items()}

        for name, mod in self.mods.items():
            flattened = {**flattened, **{name+":"+itemName: v for itemName, v in mod.report().items()}}

        return flattened

    def reset(self):
        """Clears all existing metrics. Any instantiated modules will continue
        to exist, but will be empty (it is safe to keep references to modules
        after reset()).
        """
        self.profs = {}
        for mod in self.mods.values():
            mod.reset()


# ms
timeScale = 1E3


@contextlib.contextmanager
def timer(name, timers, final=True):
    if timers is None:
        yield
    else:
        start = time.time()
        try:
            yield
        finally:
            if final:
                timers[name].increment((time.time() - start)*timeScale)
            else:
                timers[name].update((time.time() - start)*timeScale)
