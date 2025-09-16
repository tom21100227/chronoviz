from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chronoviz")
except PackageNotFoundError:
    # Fallback for local, editable installs or during development
    __version__ = "0.0.0"

__all__ = ["alignment", "plotting", "combine"]
