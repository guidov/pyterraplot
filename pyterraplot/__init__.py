from .accessor import TerraplotAccessor  # noqa: F401 — registers .tp on DataArray
from .dataset import TerraplotDatasetAccessor  # noqa: F401 — registers .tp on Dataset
from .serialize import serialize
from .server import serve
from .binary import pack_field, pack_frames
from .cog import to_cog

__version__ = "0.3.0"
__all__ = [
    "TerraplotAccessor",
    "TerraplotDatasetAccessor",
    "serialize",
    "serve",
    "pack_field",
    "pack_frames",
    "to_cog",
]
