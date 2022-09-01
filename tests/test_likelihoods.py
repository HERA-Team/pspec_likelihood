"""Test loading an UVPspec file"""
from __future__ import annotations

import astropy.units as un
import numpy as np
import pytest
from astropy import cosmology

from pspec_likelihood import (
    DataModelInterface,
    Gaussian,
    MarginalizedLinearPositiveSystematics,
)


def powerlaw_eor_spherical(z: float, k: np.ndarray, params: list[float]) -> np.ndarray:
    amplitude, index = params
    return k**3 / (2 * np.pi**2) * amplitude * un.mK**2 * (1.0 + z) / k**index


def powerlaw_eor_cylindrical(
    z: float, kcyl: tuple[np.ndarray, np.ndarray], params: list[float]
) -> np.ndarray:
    amplitude, index = params
    kperp, kpar = kcyl
    k = np.sqrt(kperp**2 + kpar**2)
    return k**3 / (2 * np.pi**2) * amplitude * un.mK**2 * (1.0 + z) / k**index


def test_like():
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(
        field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5"
    )
    dmi1 = DataModelInterface.from_uvpspec(
        uvp1,
        spw=1,
        theory_model=powerlaw_eor_spherical,
        sys_model=None,
        theory_uses_spherical_k=True,
        kpar_bins_theory=np.linspace(0.1, 1, 40) / un.Mpc,
        kperp_bins_theory=None,
        kpar_widths_theory=1e-2 * np.ones(40) / un.Mpc,
        kperp_widths_theory=None,
    )
    lk_normal = MarginalizedLinearPositiveSystematics(
        model=dmi1, set_negative_to_zero=False
    )
    lk_zeroed = MarginalizedLinearPositiveSystematics(
        model=dmi1, set_negative_to_zero=True
    )
    result_normal = lk_normal.loglike([0, -0.1], [])
    result_zeroed = lk_zeroed.loglike([0, -0.1], [])
    assert np.allclose(result_normal, -30.61810682, rtol=0, atol=1e-6), (
        "Wrong test IDR2 likelihood result",
        result_normal,
    )
    assert np.allclose(result_zeroed, -16.59249944, rtol=0, atol=1e-6), (
        "Wrong test IDR2 likelihood result",
        result_zeroed,
    )
    return result_normal, lk_zeroed


@pytest.fixture(scope="session")
def dmi_spherical():
    k = np.linspace(0.01, 0.5, 100)
    z = 9.0
    amp, indx = 1e5, 2.7
    power = powerlaw_eor_spherical(z, k, [amp, indx])
    covariance = np.diag(k**3 * 1e5)
    window_function = np.eye(len(k))

    return DataModelInterface(
        cosmology=cosmology.Planck18,
        redshift=z,
        power_spectrum=power,
        window_function=window_function,
        covariance=covariance * un.mK**4,
        kpar_bins_obs=k * cosmology.units.littleh / un.Mpc,
        theory_uses_little_h=True,
        theory_uses_spherical_k=True,
        theory_model=powerlaw_eor_spherical,
    )


@pytest.fixture(scope="session")
def dmi_cylsphere():
    kperp = np.linspace(0.01, 0.5, 15)
    kpar = np.linspace(0.01, 0.5, 100)

    kperp, kpar = np.meshgrid(kperp, kpar)
    kperp = kperp.flatten()
    kpar = kpar.flatten()
    k = np.sqrt(kpar**2 + kperp**2)

    z = 9.0
    amp, indx = 1e5, 2.7
    power = powerlaw_eor_cylindrical(z, (kperp, kpar), [amp, indx])

    covariance = np.diag(k**3 * 1e5)
    window_function = np.eye(len(k))

    return DataModelInterface(
        cosmology=cosmology.Planck18,
        redshift=z,
        power_spectrum=power,
        window_function=window_function,
        covariance=covariance * un.mK**4,
        kpar_bins_obs=kpar * cosmology.units.littleh / un.Mpc,
        kperp_bins_obs=kperp * cosmology.units.littleh / un.Mpc,
        theory_uses_little_h=True,
        theory_uses_spherical_k=True,
        theory_model=powerlaw_eor_spherical,
    )


@pytest.fixture(scope="session")
def dmi_cylindrical():
    kperp = np.linspace(0.01, 0.5, 15)
    kpar = np.linspace(0.01, 0.5, 100)

    kperp, kpar = np.meshgrid(kperp, kpar)
    kperp = kperp.flatten()
    kpar = kpar.flatten()
    k = np.sqrt(kpar**2 + kperp**2)

    z = 9.0
    amp, indx = 1e5, 2.7
    power = powerlaw_eor_cylindrical(z, (kperp, kpar), [amp, indx])

    covariance = np.diag(k**3 * 1e5)
    window_function = np.eye(len(k))

    return DataModelInterface(
        cosmology=cosmology.Planck18,
        redshift=z,
        power_spectrum=power,
        window_function=window_function,
        covariance=covariance * un.mK**4,
        kpar_bins_obs=kpar * cosmology.units.littleh / un.Mpc,
        kperp_bins_obs=kperp * cosmology.units.littleh / un.Mpc,
        theory_uses_little_h=True,
        theory_uses_spherical_k=False,
        theory_model=powerlaw_eor_cylindrical,
    )


@pytest.mark.parametrize("dmi", ["dmi_spherical", "dmi_cylsphere", "dmi_cylindrical"])
def test_max_likelihood_gaussian(request, dmi):
    dmi = request.getfixturevalue(dmi)
    like = Gaussian(model=dmi)

    amp = np.logspace(4.5, 5.5, 11)
    likes = np.zeros(len(amp))
    for i, a in enumerate(amp):
        likes[i] = like.loglike([a, 2.7], [])

    assert np.argmax(likes) == 5

    indx = np.linspace(2.6, 2.8, 11)
    likes = np.zeros(len(indx))
    for i, a in enumerate(indx):
        likes[i] = like.loglike([1e5, a], [])

    assert np.argmax(likes) == 5


@pytest.mark.parametrize("dmi", ["dmi_spherical", "dmi_cylsphere", "dmi_cylindrical"])
def test_posterior_mlp(request, dmi):
    dmi = request.getfixturevalue(dmi)
    like = MarginalizedLinearPositiveSystematics(model=dmi)

    amp = np.logspace(4.5, 5.5, 11)
    likes = np.zeros(len(amp))
    for i, a in enumerate(amp):
        likes[i] = like.loglike([a, 2.7], [])

    # For the MLPS, getting smaller and smaller amplitude gives bigger posterior,
    # since it's "more certain" to be allowed under the upper limit
    assert np.all(np.diff(likes) <= 0)
