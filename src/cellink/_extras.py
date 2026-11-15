from __future__ import annotations

import sys

# package -> (first Python version it is not available for, reason shown in the error message)
_PYTHON_LIMITS: dict[str, tuple[tuple[int, int], str]] = {
    "scooby": ((3, 13), "scooby pins pybigtools==0.1.1, which has no builds for Python 3.13 or newer"),
    "tensorqtl": (
        (3, 14),
        "tensorqtl needs pyBigWig, which has no wheels for Python 3.14 or newer",
    ),
}


# packages that are not on PyPI -> the source pip installs them from
_SOURCES: dict[str, str] = {
    "scooby": "git+https://github.com/gagneurlab/scooby.git@061b7a2275a5f8f5ffb53ae6f7f2d34e3622a7ee",
}


def install_hint(extra: str, package: str | None = None) -> str:
    """Return the pip command that installs a cellink extra, and ``package`` if it is not on PyPI."""
    command = f"pip install 'cellink-tools[{extra}]'"
    if package in _SOURCES:
        command += f" '{package} @ {_SOURCES[package]}'"
    return command


def missing_dependency_message(package: str, extra: str, purpose: str = "this function") -> str:
    """Explain how to install a missing optional dependency, or why this Python version cannot have it."""
    limit = _PYTHON_LIMITS.get(package)
    if limit is not None and sys.version_info[:2] >= limit[0]:
        (major, minor), reason = limit
        return (
            f"{package} is required for {purpose}, but it is not available for Python "
            f"{sys.version_info.major}.{sys.version_info.minor}: {reason}. "
            f"Use Python < {major}.{minor} for {package}."
        )
    return f"{package} is required for {purpose}. Install it with:\n\n    {install_hint(extra, package)}"
