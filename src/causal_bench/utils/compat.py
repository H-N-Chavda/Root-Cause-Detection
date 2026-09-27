"""Third-party compatibility shims, applied on package import.

Kept in one place so every workaround is visible and individually removable
once the upstream fix lands.
"""

from __future__ import annotations

import numpy as np

#: tigramite 5.2.10.1 calls ``np.corrcoef(..., ddof=0)`` in
#: ``independence_tests_base._get_acf``. NumPy removed ``ddof``/``bias`` in 2.0;
#: they had been deprecated *and ignored* since 1.10, so dropping them changes
#: no result -- it only stops the TypeError. Without this, any CMIknn test with
#: an empty conditioning set raises, which is every test in PCMCI+'s first
#: phase. Remove once tigramite > 5.2.10.1 drops the argument.
_CORRCOEF_PATCHED = False


def patch_numpy_corrcoef() -> bool:
    """Strips the dead ``ddof``/``bias`` arguments from ``np.corrcoef``.

    Returns True if the patch was applied, False if it was already in place or
    the installed NumPy still accepts the arguments.
    """
    global _CORRCOEF_PATCHED
    if _CORRCOEF_PATCHED:
        return False
    try:
        np.corrcoef(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]), ddof=0)
    except TypeError:
        pass
    else:
        return False  # NumPy still accepts it; nothing to do.

    original = np.corrcoef

    def corrcoef(*args: object, **kwargs: object):
        kwargs.pop("ddof", None)
        kwargs.pop("bias", None)
        return original(*args, **kwargs)

    corrcoef.__doc__ = original.__doc__
    np.corrcoef = corrcoef  # type: ignore[assignment]
    _CORRCOEF_PATCHED = True
    return True
