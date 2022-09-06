"""Types to be used throughout the package."""
from __future__ import annotations

from typing import Any, Callable

import attr
import numpy as np
from astropy import units as u

PowerType = u.Quantity[u.mK**2]
CovarianceType = u.Quantity[u.mK**4]
Wavenumber = u.Quantity["wavenumber"]

cmp_array = attr.cmp_using(eq=np.array_equal)


def is_unit(unit: str) -> bool:
    """Evaluate whether a given string corresponds to an existing astropy unit."""
    if isinstance(unit, u.Unit):
        return True

    try:
        u.Unit(unit)
        return True
    except ValueError:
        return False


def vld_unit(
    unit: str | u.Unit, equivalencies=()
) -> Callable[[Any, attr.Attribute, Any], None]:
    """Attr validator to check physical type."""
    utype = is_unit(unit)
    if not utype:
        # must be a physical type. This errors with ValueError if unit is not
        # really a physical type.
        u.get_physical_type(unit)

    def _check_type(self, att: attr.Attribute, val: Any):
        if not isinstance(val, u.Quantity):
            raise TypeError(f"{att.name} must be an astropy Quantity!")

        if utype and not val.unit.is_equivalent(unit, equivalencies):
            raise u.UnitConversionError(
                f"{att.name} not convertible to {unit}. Got {val.unit}"
            )

        if not utype and val.unit.physical_type != unit:
            raise u.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. "
                f"Got '{val.unit.physical_type}'"
            )

    return _check_type
