"""Python Quantum Information Toolkit

See the README.rst file included in this distribution,
or the project website at http://qit.sourceforge.net/
"""
from importlib.metadata import version, PackageNotFoundError

import scipy.constants as const

from .base import *
from .lmap import *
from .utils import *
from .state import *
from .plot import *
from . import gate, hamiltonian, ho, invariant, markov, seq, examples


# find out the version number of the installed dist package
try:
    __version__ = version('qit')
except PackageNotFoundError:
    __version__ = 'unknown'
finally:
    del version, PackageNotFoundError


def version():
    """Returns the QIT version number (as a string)."""
    return __version__
