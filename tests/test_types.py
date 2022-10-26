import attr
import pytest
from astropy import units as un

from pspec_likelihood import types as tp


def test_is_unit():
    assert tp.is_unit(un.mK)
    assert tp.is_unit("mK")
    assert not tp.is_unit("not_a_unit")


def test_vld_unit():
    with pytest.raises(ValueError):
        tp.vld_unit("super_terrible")

    @attr.define
    class _A:
        temp_field = attr.ib(validator=tp.vld_unit("temperature"))

    with pytest.raises(TypeError, match="temp_field must be an astropy Quantity!"):
        _A(temp_field=3)

    with pytest.raises(
        un.UnitConversionError,
        match="temp_field must have physical type of 'temperature'",
    ):
        _A(temp_field=3 * un.m)

    a = _A(3 * un.mK)
    assert a.temp_field.unit == un.mK
    a = _A(3 * un.K)
    assert a.temp_field.unit == un.K

    @attr.define
    class _B:
        temp_field = attr.ib(validator=tp.vld_unit(un.mK))

    with pytest.raises(
        un.UnitConversionError, match="temp_field not convertible to mK"
    ):
        _B(temp_field=3 * un.m)

    b = _B(3 * un.mK)
    assert b.temp_field.unit == un.mK

    b = _B(3 * un.K)
    assert b.temp_field.unit == un.K
