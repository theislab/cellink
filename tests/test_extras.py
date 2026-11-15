import collections
import sys

import pytest

from cellink import _extras
from cellink._extras import install_hint, missing_dependency_message

_Version = collections.namedtuple("_Version", "major minor micro releaselevel serial")


def test_install_hint_names_the_pypi_distribution():
    assert install_hint("pgen") == "pip install 'cellink-tools[pgen]'"
    assert install_hint("pgen", "pgenlib") == "pip install 'cellink-tools[pgen]'"


def test_install_hint_adds_packages_that_are_not_on_pypi():
    assert install_hint("scooby", "scooby") == (
        "pip install 'cellink-tools[scooby]' 'scooby @ git+https://github.com/gagneurlab/scooby.git@061b7a2275a5f8f5ffb53ae6f7f2d34e3622a7ee'"
    )


@pytest.mark.parametrize(
    ("package", "version", "unavailable"),
    [
        ("scooby", (3, 12), False),
        ("scooby", (3, 13), True),
        ("tensorqtl", (3, 13), False),
        ("tensorqtl", (3, 14), True),
    ],
)
def test_message_says_why_a_python_version_cannot_have_the_package(monkeypatch, package, version, unavailable):
    monkeypatch.setattr(sys, "version_info", _Version(*version, 0, "final", 0))
    message = missing_dependency_message(package, package, "`f`")
    if unavailable:
        assert f"not available for Python {version[0]}.{version[1]}" in message
        assert "pip install" not in message
    else:
        assert message.endswith(install_hint(package, package))


def test_packages_without_limits_get_the_install_hint(monkeypatch):
    monkeypatch.setattr(sys, "version_info", _Version(3, 15, 0, "final", 0))
    assert missing_dependency_message("scdrs", "scdrs").endswith(install_hint("scdrs"))
    assert "scdrs" not in _extras._PYTHON_LIMITS


def test_scooby_runner_reports_the_python_limit(monkeypatch):
    from cellink.tl.external._scooby import ScoobyRunner

    monkeypatch.setitem(sys.modules, "scooby.modeling", None)  # import fails as if scooby were absent
    monkeypatch.setattr(sys, "version_info", _Version(3, 13, 0, "final", 0))
    with pytest.raises(ImportError, match="not available for Python 3.13"):
        ScoobyRunner().get_scooby_class()
