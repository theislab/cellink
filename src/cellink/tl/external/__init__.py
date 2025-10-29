# from ._jaxqtl import run_jaxqtl, read_jaxqtl_results
# from ._ld import calculate_ld
# from ._mixmil import run_mixmil
# from ._pc import calculate_pcs
# from ._tensorqtl import run_tensorqtl, read_tensorqtl_results

# # src/cellink/tl/external/__init__.py

# Use a specific list of attributes to control what is exposed
__all__ = [
    "run_jaxqtl",
    "read_jaxqtl_results",
    "calculate_ld",
    "run_mixmil",
    "calculate_pcs",
    "run_tensorqtl",
    "read_tensorqtl_results",
]


# Define lazy import getters
def __getattr__(name):
    # This function is called only when an attribute (like run_jaxqtl) is accessed
    if name in __all__:
        if name in ("run_jaxqtl", "read_jaxqtl_results"):
            from ._jaxqtl import read_jaxqtl_results, run_jaxqtl

            return locals()[name]
        elif name == "calculate_ld":
            from ._ld import calculate_ld

            return calculate_ld
        elif name == "run_mixmil":
            from ._mixmil import run_mixmil

            return run_mixmil
        elif name == "calculate_pcs":
            from ._pc import calculate_pcs

            return calculate_pcs
        elif name in ("run_tensorqtl", "read_tensorqtl_results"):
            from ._tensorqtl import read_tensorqtl_results, run_tensorqtl

            return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
