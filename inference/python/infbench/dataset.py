import abc


class processor():
    """Container for transformations over a dataset. Doesn't hold any complex
    refernces, it's really just a namespace.

    Methods may return multiple values, some or all of these values may be
    needed by subsequent steps. To track how things plug together, every
    dataset must have self.preMap and self.postMap fields. These are maps that
    list the indices of the previous steps that they want to consume (in
    order):
        nOutputPre: Number of return values of pre function
        nOutputPost: Number of return values of post function
        preMap:  tuple of indices of output from get
        postMap: Which outputs to take from pre, the model may only output one
                 value and that will be the last input given.
    """

    def pre(self, dat):
        pass

    def post(self, dat):
        pass


class loader(abc.ABC):
    """Handle to a dataset, used for reading inputs. Does not pre or post
    process data at all."""

    @property
    @abc.abstractmethod
    def ndata(self):
        pass

    @property
    @abc.abstractmethod
    def checkAvailable(self):
        """Is the check() function defined?"""
        pass

    @abc.abstractmethod
    def preLoad(self, idxs):
        """Some datasets are too big to fit in RAM in their entirety, preload
        will load a subset of data based on the indexes in idxs"""
        pass

    @abc.abstractmethod
    def unLoad(self, idxs):
        """Free memory associated with idxs"""
        pass

    @abc.abstractmethod
    def get(self, idx):
        """Returns a single datum at idx. Datum is of a dataset-dependent type."""
        pass

    @abc.abstractmethod
    def check(self, result, idx):
        """Check if the result of index idx is correct"""
        pass
