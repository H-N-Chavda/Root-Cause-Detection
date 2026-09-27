"""causal_bench - PC causal discovery on the CIPCaD-Bench industrial datasets.

The package name is defined once, in `paths.PACKAGE_NAME`.
"""

from .paths import PACKAGE_NAME
from .utils.compat import patch_numpy_corrcoef

# Applied on import: tigramite needs it before any CMIknn test runs.
patch_numpy_corrcoef()

__all__ = ["PACKAGE_NAME", "__version__"]

__version__ = "0.1.0"
