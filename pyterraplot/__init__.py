from .accessor import TerraplotAccessor  # noqa: F401 — registers .tp on DataArray
from .serialize import serialize
from .server import serve

__version__ = "0.1.0"
__all__ = ["TerraplotAccessor", "serialize", "serve"]
