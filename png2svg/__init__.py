from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("png2svg")
except PackageNotFoundError:  # pragma: no cover - when not installed
    __version__ = "0.1.0"

