"""
XAgent - Robot Agent Brain for task planning and control
"""

__version__ = "0.2.0"
__author__ = "AgileX Team"

from .xagent import XAgent
from .xbrain import Xbrain

__all__ = ["XAgent", "Xbrain", "__version__"]