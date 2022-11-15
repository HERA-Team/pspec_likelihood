"""Test loading an UVPspec file"""
from __future__ import annotations

import astropy.units as un
import attr
import numpy as np
import pytest
from astropy import cosmology
from astropy.cosmology import Planck18

from pspec_likelihood import (
    DataModelInterface,
    Gaussian,
    LikelihoodLinearSystematic,
    MarginalizedLinearPositiveSystematics,
)


def powerlaw_eor_spherical(z: float, k: np.ndarray, params: list[float]) -> np.ndarray:
    amplitude, index = params
    return k**3 / (2 * np.pi**2) * amplitude * un.mK**2 * (1.0 + z) / k**index


def powerlaw_eor_spherical_littleh(
    z: float, k: np.ndarray, params: list[float]
) -> np.ndarray:
    amplitude, index = params
    return (
        (k * Planck18.h) ** 3
        / (2 * np.pi**2)
        * amplitude
        * un.mK**2
        * (1.0 + z)
        / (k * Planck18.h) ** index
    )


def powerlaw_eor_cylindrical(
    z: float, kcyl: tuple[np.ndarray, np.ndarray], params: list[float]
) -> np.ndarray:
    amplitude, index = params
    kperp, kpar = kcyl
    k = np.sqrt(kperp**2 + kpar**2)
    return k**3 / (2 * np.pi**2) * amplitude * un.mK**2 * (1.0 + z) / k**index


def test_like(uvp1):
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
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
    with pytest.warns(UserWarning, match="Ignoring data in positions"):
        result_normal = lk_normal.loglike([0, -0.1], [])

    with pytest.warns(UserWarning, match="Ignoring data in positions"):
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


def test_little_h(uvp1):
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    dmi1 = DataModelInterface.from_uvpspec(
        uvp1,
        spw=1,
        theory_model=powerlaw_eor_spherical,
        theory_uses_little_h=False,
        sys_model=None,
        theory_uses_spherical_k=True,
        kpar_bins_theory=np.linspace(0.1, 1, 40) / un.Mpc,
        kperp_bins_theory=None,
        kpar_widths_theory=1e-2 * np.ones(40) / un.Mpc,
        kperp_widths_theory=None,
    )
    dmi2 = DataModelInterface.from_uvpspec(
        uvp1,
        spw=1,
        theory_model=powerlaw_eor_spherical_littleh,
        theory_uses_little_h=True,
        sys_model=None,
        theory_uses_spherical_k=True,
        kpar_bins_theory=np.linspace(0.1, 1, 40) / un.Mpc,
        kperp_bins_theory=None,
        kpar_widths_theory=1e-2 * np.ones(40) / un.Mpc,
        kperp_widths_theory=None,
    )

    with pytest.warns(UserWarning, match="Your covariance matrix is not diagonal"):
        lk_no_littleh = MarginalizedLinearPositiveSystematics(
            model=dmi1, set_negative_to_zero=False
        )
    with pytest.warns(UserWarning, match="Your covariance matrix is not diagonal"):
        lk_with_littleh = MarginalizedLinearPositiveSystematics(
            model=dmi2, set_negative_to_zero=False
        )

    assert np.isclose(
        lk_no_littleh.loglike([1000.0, 2.1], []),
        lk_with_littleh.loglike([1000.0, 2.1], []),
        rtol=1e-3,
    )


# theory_uses_little_h is False by default


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


@pytest.mark.parametrize("dmi", ["dmi_spherical", "dmi_cylsphere", "dmi_cylindrical"])
def test_max_likelihood_arblinearsystematics(request, dmi):
    dmi = request.getfixturevalue(dmi)
    like = LikelihoodLinearSystematic(
        linear_systematics_basis_function=linear_systematics_basis_function,
        nlinear=1,
        model=dmi,
    )

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
def test_arblin_gauss_vs_uniform(request, dmi):
    dmi = request.getfixturevalue(dmi)

    like_unif = LikelihoodLinearSystematic(
        linear_systematics_basis_function=linear_systematics_basis_function,
        nlinear=1,
        model=dmi,
    )
    lu = like_unif.loglike([4.5, 2.7], [])

    lg = []
    for i, sigma in enumerate([1e3, 1e4, 1e5, 1e6]):
        like_gauss = LikelihoodLinearSystematic(
            linear_systematics_basis_function=linear_systematics_basis_function,
            mu_theta=np.array([0]),
            sigma_theta=np.array([[sigma]]),
            model=dmi,
        )
        lg.append(like_gauss.loglike([4.5, 2.7], []))

        if i:
            assert np.abs(lu - lg[i]) <= np.abs(lu - lg[i - 1])


def test_arblin_bad_inputs(dmi_spherical):
    with pytest.raises(ValueError, match="You need to provide nlinear"):
        LikelihoodLinearSystematic(
            linear_systematics_basis_function=linear_systematics_basis_function,
            model=dmi_spherical,
        )

    with pytest.raises(ValueError, match="Provide sigma_theta"):
        LikelihoodLinearSystematic(
            linear_systematics_basis_function=linear_systematics_basis_function,
            mu_theta=np.array([0]),
            model=dmi_spherical,
        )
    with pytest.raises(ValueError, match="Covariance must be two dimensional"):
        LikelihoodLinearSystematic(
            linear_systematics_basis_function=linear_systematics_basis_function,
            sigma_theta=np.array([1]),
            mu_theta=np.array([0]),
            model=dmi_spherical,
        )

    with pytest.raises(ValueError, match="Covariance must be square"):
        LikelihoodLinearSystematic(
            linear_systematics_basis_function=linear_systematics_basis_function,
            sigma_theta=np.array([[1, 1]]),
            mu_theta=np.array([0]),
            model=dmi_spherical,
        )

    with pytest.raises(ValueError, match="Covariance is not invertible"):
        LikelihoodLinearSystematic(
            linear_systematics_basis_function=linear_systematics_basis_function,
            sigma_theta=np.array([[0]]),
            mu_theta=np.array([0]),
            model=dmi_spherical,
            cov_tolerance=0,
        )

    def bad_sys_function(sys_params, kperp_bins_obs, kpar_bins_obs):
        return np.ones((len(kpar_bins_obs), 1))

    lk = LikelihoodLinearSystematic(
        linear_systematics_basis_function=bad_sys_function,
        nlinear=1,
        model=dmi_spherical,
    )

    with pytest.raises(ValueError, match="must return a power-like quantity"):
        lk.loglike([4, 2.7], [])

    def bad_sys_function(sys_params, kperp_bins_obs, kpar_bins_obs):
        return np.ones((len(kpar_bins_obs) + 1, 1)) << un.mK**2

    lk = LikelihoodLinearSystematic(
        linear_systematics_basis_function=bad_sys_function,
        nlinear=1,
        model=dmi_spherical,
    )

    with pytest.raises(ValueError, match="must return a "):
        lk.loglike([4, 2.7], [])


def test_different_discretization(dmi_spherical):
    bin_widths = dmi_spherical.kpar_bins_obs[1:] - dmi_spherical.kpar_bins_obs[:-1]
    bin_widths = np.concatenate(([bin_widths[0]], bin_widths))
    dmi_trapz = attr.evolve(
        dmi_spherical, window_integration_rule="trapz", kpar_widths_theory=bin_widths
    )

    dmi_quad = attr.evolve(
        dmi_spherical, window_integration_rule="quad", kpar_widths_theory=bin_widths
    )

    centre = dmi_spherical.compute_model([5.0, 2.7], [])
    trapz = dmi_trapz.compute_model([5.0, 2.7], [])
    quad = dmi_quad.compute_model([5.0, 2.7], [])

    assert np.allclose(centre, trapz, atol=1e-2)
    assert np.allclose(centre, quad, atol=1e-2)


def constant_offset_systematic(z: float, k: np.ndarray, params: list[float]):
    return params[0] * np.ones(len(k)) * un.mK**2


def test_with_sys_model(dmi_spherical):
    new_dmi = attr.evolve(dmi_spherical, sys_model=constant_offset_systematic)

    assert np.array_equal(
        new_dmi.compute_model([5.0, 2.7], [0]),
        dmi_spherical.compute_model([5.0, 2.7], []),
    )

    assert not np.array_equal(
        new_dmi.compute_model([5.0, 2.7], [1.0]),
        dmi_spherical.compute_model([5.0, 2.7], []),
    )


def test_with_sys_model_not_apply_window(dmi_spherical):
    new_dmi = attr.evolve(
        dmi_spherical,
        sys_model=constant_offset_systematic,
        apply_window_to_systematics=False,
    )

    assert np.array_equal(
        new_dmi.compute_model([5.0, 2.7], [0]),
        dmi_spherical.compute_model([5.0, 2.7], []),
    )

    assert not np.array_equal(
        new_dmi.compute_model([5.0, 2.7], [1.0]),
        dmi_spherical.compute_model([5.0, 2.7], []),
    )


def powerlaw_eor_spherical_dictparams(
    z: float, k: np.ndarray, params: dict[str, float]
) -> np.ndarray:
    return (
        k**3
        / (2 * np.pi**2)
        * params["amplitude"]
        * un.mK**2
        * (1.0 + z)
        / k ** params["index"]
    )


def test_with_paramnames(dmi_spherical):
    dmi_names = attr.evolve(
        dmi_spherical,
        theory_param_names=["amplitude", "index"],
        theory_model=powerlaw_eor_spherical_dictparams,
    )

    assert np.array_equal(
        dmi_names.compute_model({"amplitude": 5.0, "index": 2.7}, []),
        dmi_spherical.compute_model([5.0, 2.7], []),
    )


def linear_systematics_basis_function(sys_params, kperp_bins_obs, kpar_bins_obs):
    return np.ones((len(kpar_bins_obs), 1)) << un.mK**2
