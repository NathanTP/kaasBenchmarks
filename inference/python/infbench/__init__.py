
# Someone imports tensorflow somewhere and it logs a bunch of junk, this
# suppresses it
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from . import dataset
from . import model
