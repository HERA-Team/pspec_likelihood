"""The PSpec Likelihood Package."""

from importlib.metadata import PackageNotFoundError, version

from . import likelihood as likelihood
from .arbitrary_linear_systematics import LikelihoodLinearSystematic as LikelihoodLinearSystematic
from .likelihood import (
    DataModelInterface as DataModelInterface,
)
from .likelihood import (
    Gaussian as Gaussian,
)
from .likelihood import (
    GaussianLinearSystematics as GaussianLinearSystematics,
)
from .likelihood import (
    MarginalizedLinearPositiveSystematics as MarginalizedLinearPositiveSystematics,
)

try:
    __version__ = version("pspec_likelihood")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
