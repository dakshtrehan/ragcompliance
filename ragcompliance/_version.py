"""Single source of truth for the package version.

``pyproject.toml`` reads this file through hatch's dynamic version
support, and ``ragcompliance.__version__`` re-exports it, so the number
only ever lives here.
"""

__version__ = "1.0.0rc1"
