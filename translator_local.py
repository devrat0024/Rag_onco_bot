# Compatibility shim: some modules import `translator_local` while the canonical file is `translator.py`.
# This file simply re-exports LocalTranslator from translator.py so existing imports work.
try:
    from translator import LocalTranslator
except Exception:
    # In rare cases the package context may require a relative import
    from .translator import LocalTranslator  # type: ignore

__all__ = ["LocalTranslator"]
