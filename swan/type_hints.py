from pathlib import Path
from typing import Union, Any, List, TYPE_CHECKING
import numpy as np

PathLike = Union[str, Path]

if TYPE_CHECKING:
    import numpy.typing as npt
    ArrayLike = npt.ArrayLike
else:
    ArrayLike = Union[List[Any], np.ndarray]
