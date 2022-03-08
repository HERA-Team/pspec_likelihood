"""Primary module defining likelihoods based on HERA power spectra."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, Sequence

import astropy.cosmology as csm
import astropy.units as un
import attr
import numpy as np
from cached_property import cached_property
from hera_pspec import grouping
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from . import types as tp
from .types import vld_unit


@attr.s(kw_only=True)
class DataModelInterface:
    r"""Container for power spectrum measurements and models.

    This class keeps track of power-spectrum measurements
    (and their associated covariances and window functions)
    along with a theoretical model and calculations of the likelihoods
    given this model that propertly account for the window functions.
    For now, this container assumes Gaussian measurement errors and
    thus only keeps track of covariances but this may change in the future.

    Parameters
    ----------
    cosmology
        The :class:`astropy.cosmology.FLRW` object defining the cosmology.
    power_spectrum
        The 1D or 2D power spectrum values of the observation in squared temperature
        units. Whether 1D or 2D, this array is 1D (i.e. flattened).
    window_function
        The window function that takes theory space to observational space. Must be an
        array whose first dimension has length equal to the power spectrum, and the
        second dimension has the same length as
        ``kpar_bins_theory``/``kperp_bins_theory``.
    covariance
        The data covariance matrix. If 2D, must be a square matrix with each dimension
        the same length as ``power_spectrum``. If 1D, the covariance is assumed to be
        diagonal, and must have the same length as ``power_spectrum``.
    kperp_bins_obs
        The k-perpendicular bins of the observation. These provide the bin *centres*,
        and so there should be the same number as power spectrum values.
        If ``kpar_bins_obs`` is not provided, ``kperp_bins_obs`` is treated as the
        spherically-averaged k bins.
    kpar_bins_obs
        If provided, the k-parallel bins of the observation.
        See notes on ``kperp_bins_obs``. If not provided, treat ``kperp_bins_obs``
        as spherical.
    kperp_bins_theory
        The k-perpendicular bins of the theory. These provide the bin *centres*, and
        so there should be the same number as the second dimension of the window
        function.
        If ``kpar_bins_theory`` is not provided, ``kperp_bins_theory`` is treated as the
        spherically-averaged k bins.
    kpar_bins_theory
        If provided, the k-parallel bins of the theory.
        See notes on ``kperp_bins_theory``. If not provided, treat ``kperp_bins_theory``
        as spherical.
    kperp_widths_theory
        If provided, the k-perpendicular bin widths associated with each bin. It is
        assumed that the bins are in the (linear-space) centre. Only required if
        ``window_integration_rule`` is not ``"midpoint"``.
    kpar_widths_theory
        If provided, the k-parallel bin widths associated with each bin. It is
        assumed that the bins are in the (linear-space) centre. Only required if
        ``window_integration_rule`` is not ``"midpoint"``.
    window_integration_rule
        Converting from theory space to observational space is done by integrating the
        ``window_function * theory_function`` over k-space for a given choice of
        observational co-ordinates. This integral is discretized, and this parameter
        provides the rule by which the *theory function* portion of the integral is
        discretized. Choices are 'midpoint', 'trapz' or 'quad'.
    theory_uses_little_h
        Whether the theory function accepts wavenumbers in units of h/Mpc. If False,
        it accepts wavenumbers in units of 1/Mpc.
    theory_uses_spherical_k
        If True, the theory function only accepts spherical k, rather than cylindrical
        k.
    theory_model
        The callable theoretical power spectrum as a function of redshift and k.
        The signature of the function should be ``theory(z, k, params) -> p``,
        where `z` is a float redshift,and `params` is a tuple of float parameters OR a
        dict of float parameters. If it takes a dict of parameters,
        ``theory_param_names`` must be given. ``k`` is either a 2-tuple
        ``(kperp, kpar)``, where each of these arrays is 1D and the same length, or,
        if ``theory_uses_spherical_k`` is True, a single array of ``|k|``.
        The output ``p`` is an array of the same length as ``kperp`` and ``kpar``
        (or simply ``k``), containing the power spectrum in mK^2.
    theory_param_names
        If given, pass a dictionary of parameters to the ``theory_model`` whose keys
        are the names.
    apply_window_to_systematics
        Whether the systematics are defined in theory-space or data-space. If defined
        in theory-space, the window function must be applied to the resulting
        systematics.
    """

    redshift: float = attr.ib(converter=float)
    power_spectrum: tp.PowerType = attr.ib(
        validator=vld_unit(un.mK**2), eq=tp.cmp_array
    )
    window_function: np.ndarray = attr.ib(eq=tp.cmp_array, converter=np.array)
    covariance: tp.CovarianceType = attr.ib(validator=vld_unit(un.mK**4))
    theory_model: Callable = attr.ib(validator=attr.validators.is_callable())
    sys_model: Callable = attr.ib(validator=attr.validators.is_callable())

    cosmology: csm.FLRW = attr.ib(
        csm.Planck18, validator=attr.validators.instance_of(csm.FLRW)
    )
    kperp_bins_obs: tp.Wavenumber = attr.ib()
    kpar_bins_obs: tp.Wavenumber | None = attr.ib()
    kperp_bins_theory: tp.Wavenumber = attr.ib()
    kpar_bins_theory: tp.Wavenumber = attr.ib()
    kperp_widths_theory: tp.Wavenumber | None = attr.ib()
    kpar_widths_theory: tp.Wavenumber | None = attr.ib()

    window_integration_rule: Literal["midpoint", "trapz", "quad"] = attr.ib(
        "midpoint", validator=attr.validators.in_(("midpoint", "trapz", "quad"))
    )

    theory_uses_little_h: bool = attr.ib(default=False, converter=bool)
    theory_uses_spherical_k: bool = attr.ib(default=False, converter=bool)

    theory_param_names: Sequence[str] | None = attr.ib(
        None, converter=attr.converters.optional(tuple)
    )
    sys_param_names: Sequence[str] | None = attr.ib(
        None, converter=attr.converters.optional(tuple)
    )
    apply_window_to_systematics: bool = attr.ib(True, converter=bool)

    @kperp_bins_obs.validator
    @kperp_bins_theory.validator
    def _k_validator(self, att, val):
        if not np.isrealobj(val):
            raise TypeError(f"{att.name} must be real!")

        tp.vld_unit("wavenumber", equivalencies=un.with_H0(self.cosmology.H0))

        if val.shape != self.power_spectrum.shape:
            raise ValueError(f"{att.name} must have same shape as the power spectrum")

    @kpar_bins_obs.validator
    @kpar_bins_theory.validator
    @kpar_widths_theory.validator
    @kperp_widths_theory.validator
    def _opt_k_vld(self, att, val):
        if val is not None:
            self._k_validator(att, val)

    @window_function.validator
    def _wf_vld(self, att, val):
        if val.shape not in [(len(self.power_spectrum), len(self.power_spectrum))]:
            raise ValueError("window_function must be  Nk * Nk matrix")

    @window_integration_rule.validator
    def _wir_vld(self, att, val):
        if (
            val != "midpoint"
            and self.kpar_widths_theory is None
            or self.kperp_widths_theory is None
        ):
            raise ValueError(
                f"if window_integration_rule={val}, kpar/kperp widths are required."
            )

    @covariance.validator
    def _cov_vld(self, att, val):
        if val.shape not in [
            (len(self.power_spectrum),),
            (len(self.power_spectrum), len(self.power_spectrum)),
        ]:
            raise ValueError("covariance must be Nk*Nk matrix or length-Nk vector")

    @cached_property
    def spherical_kbins_obs(self) -> tp.Wavenumber:
        """The spherical k bins of the observation (the edges)."""
        if self.kpar_bins_obs is not None:
            return np.sqrt(self.kpar_bins_obs**2 + self.kperp_bins_obs**2)
        else:
            return self.kperp_bins_obs

    @cached_property
    def spherical_kbins_theory(self) -> tp.Wavenumber:
        """The spherical k bins of the theory (edges)."""
        if self.kpar_bins_theory is not None:
            return np.sqrt(self.kpar_bins_theory**2 + self.kperp_bins_theory**2)
        else:
            return self.kperp_bins_theory

    @cached_property
    def kperp_centres(self) -> tp.Wavenumber:
        """Centres of the kperp bins."""
        return (self.kperp_bins_obs[1:] + self.kperp_bins_obs[:-1]) / 2

    @classmethod
    def from_uvpspec(
        cls, uvpspec, theory_uses_spherical_k: bool = False, **kwargs
    ) -> DataModelInterface:
        """Construct the ModelDataInterface from a UVPSpec object."""
        if theory_uses_spherical_k:
            grouping.spherical_average(uvpspec, **kwargs)

        raise NotImplementedError("Not yet, sorry!")

    def _kconvert(self, k):
        return k.to_value(
            "littleh/Mpc" if self.theory_uses_little_h else "1/Mpc",
            equivalencies=un.with_H0(self.cosmology.H0),
        )

    def _discretize(
        self,
        model: Callable,
        z: float,
        k: tuple[np.ndarray, np.ndarray] | np.ndarray,
        kwidth: tuple[np.ndarray, np.ndarray] | np.ndarray,
        params: Sequence[float] | dict[str, float],
    ) -> tuple[tp.PowerType, tp.PowerType]:
        if self.window_integration_rule == "midpoint":
            results = model(z, k, params)
            errors = None
        elif self.window_integration_rule == "trapz":
            lower = model(z, k - kwidth / 2, params)
            upper = model(z, k + kwidth / 2, z, params)
            results = (lower + upper) / 2
            errors = (lower - upper) / 2
        elif self.window_integration_rule == "quad":

            def pk_func(k):
                return model(z, k, params)

            results = []
            errors = []
            for center, width in zip(k, kwidth):
                result, error = quad(pk_func, center - width / 2, center + width / 2)
                results.append(result / width)
                errors.append(error / width)

        return results, errors

    def discretize_theory(
        self, z: float, theory_params: Sequence[float] | dict[str, float]
    ) -> tuple[tp.PowerType, tp.PowerType]:
        r"""Compute the discretized power spectrum at the (k, z) of the window function.

        This outputs an approximation of the integral of the theory power spectrum over
        each cylindrical k-bin. The way in which this is done is controlled by
        :attr:`window_integration_rule`.

        Parameters
        ----------
        z
            The redshift
        theory_params
            sequence of parameters passed to the theory function

        Returns
        -------
        results
            list of power spectrum values corresponding to the bins
        errors
            Estimation of the error through binning, if a suitable method has been
            chosen, otherwise None.
        """
        if self.theory_uses_spherical_k:
            k = self._kconvert(self.spherical_kbins_theory)
            if self.window_integration_rule != "midpoint":
                kwidth = self._kconvert(self.spherical_kbins_width_theory)

        else:
            k = (
                self._kconvert(self.kperp_bins_theory),
                self._kconvert(self.kpar_bins_theory),
            )
            kwidth = 0  # TODO: need to do this correctly.

        theory_params = self._validate_params(theory_params, self.theory_param_names)
        return self._discretize(self.theory_model, z, k, kwidth, theory_params)

    def discretize_systematics(
        self, z: float, sys_params: Sequence[float] | dict[str, float]
    ) -> tuple[tp.PowerType, tp.PowerType]:
        r"""Compute the discretized systematic power.

        This outputs an approximation of the integral of the theory power spectrum over
        each cylindrical k-bin. The way in which this is done is controlled by
        :attr:`window_integration_rule`.
        """
        if self.apply_window_to_systematics:
            k = (
                self._kconvert(self.kperp_bins_theory),
                self._kconvert(self.kpar_bins_theory),
            )
        else:
            k = (
                self._kconvert(self.kperp_bins_obs),
                self._kconvert(self.kpar_bins_obs),
            )

        sys_params = self._validate_params(sys_params, self.sys_param_names)
        return self._discretize(self.sys_model, z, k, sys_params)

    def apply_window_function(self, discretized_model: tp.PowerType) -> tp.PowerType:
        r"""Calculate theoretical power spectrum with data window function applied.

        Simply performs the matrix product :math:`p_w = W p_m`.

        Parameters
        ----------
        discretized_theory
            The discretized theoretical power spectrum, discretized in such a way as to
            be compatible with the window function. This can be calculated with
            :meth:`discretize_theory`.

        Returns
        -------
        windowed_theory
            A 1D array of power-spectrum values corresponding to :attr:`kpar_bins_obs``.
        """
        return self.window_function.dot(discretized_model)

    def _validate_params(self, params, names) -> tuple[float] | dict[str, float]:
        r"""
        Check if params is a list or dictionary.

        If list convert to dictionary using param_names.

        Parameters
        ----------
        params : dictionary, list or tuple

        Returns
        -------
        params : dictionary
            params convert to dictionary
        """
        if isinstance(params, dict):
            if names is not None:
                assert set(names) == set(
                    params.keys()
                ), "input parameters don't match parameters of the likelihood"
            return params
        else:
            if names is None:
                return tuple(params)

            if len(names) != len(params):
                raise ValueError(
                    "input params is not of same length as given param_names "
                    f"({len(names)} vs {len(params)})"
                )
            return dict(zip(names, params))

    def compute_model(
        self,
        theory_params: Sequence[float] | dict[str, float],
        sys_params: Sequence[float] | dict[str, float],
    ):
        """Compute the theory+systematics model in the data-space."""
        theory, error = self.discretize_theory(self.redshift, theory_params)
        sys, error = self.discretize_systematics(self.redshift, sys_params)

        if self.apply_window_to_systematics:
            return self.apply_window_function(theory + sys)
        else:
            return self.apply_window_function(theory) + sys


@attr.s(kw_only=True)
class PSpecLikelihood(ABC):
    """Base class for likelihoods.

    The base class implements one abstract method: :meth:`loglike`, which must compute
    the floating-point log-likelihood given the theory and systematics parameters that
    are being sampled/varied.

    Parameters
    ----------
    model
        An instance of :class:`DataModelInterface` that is used to do the transformation
        from theory to data space.
    """

    model: DataModelInterface = attr.ib()

    @abstractmethod
    def loglike(self, theory_params, sys_params) -> float:
        """Compute the log-likelihood."""
        pass


@attr.s(kw_only=True)
class Gaussian(PSpecLikelihood):
    """The simplest Gaussian likelihood."""

    def loglike(self, theory_params, sys_params) -> float:
        """Compute the log likelihood."""
        model = self.model.compute_model(theory_params, sys_params)
        normal = multivariate_normal(
            mean=self.model.power_spectrum, cov=self.model.covariance
        )
        return normal.logpdf(model)


@attr.s(kw_only=True)
class GaussianLinearSystematics(PSpecLikelihood):
    """A Gaussian likelihood where some systematics are assumed to be linear.

    Parameters
    ----------
    linear_systematics_basis_function
        A function that, given a set of non-linear systematics parameters (potentially
        an empty set), will compute the basis set corresponding to the known linear
        parameters of the model, at the kperp/kpar of either the observation or theory.
    linear_systematics_mean
        The prior mean of the linear systematics.
    linear_systematics_cov
        The prior covariance of the linear systematics.
    """

    linear_systematics_basis_function: Callable = attr.ib()
    linear_systematics_mean: np.ndarray = attr.ib()
    linear_systematics_cov: np.ndarray = attr.ib()

    def get_mu_linear(self, basis: np.ndarray) -> tuple[float]:
        """Compute the posterior mean of linear parameters."""
        pass

    def get_sigma_linear(self, basis) -> tuple[float]:
        """Compute the posterior covariance of the linear parameters."""
        pass

    def loglike(self, theory_params, sys_params) -> float:
        """Compute the log likelihood."""
        # Here we get only the non-linera sys params, but the model requires all of
        # them. "basis" corresponds to "A" in the memo.
        basis = self.linear_systematics_basis_function(
            sys_params,
            self.model.kperp_bins_theory
            if self.model.apply_window_to_systematics
            else self.model.kperp_bins_obs,
            self.model.kpar_bins_theory
            if self.model.apply_window_to_systematics
            else self.model.kpar_bins_obs,
        )

        mu_linear = self.get_mu_linear(basis)
        sig_linear = self.get_sigma_linear(basis)

        model = self.model.compute_model(theory_params, tuple(sys_params) + mu_linear)
        normal = multivariate_normal(
            mean=self.model.power_spectrum, cov=self.model.covariance
        )

        prior = multivariate_normal(
            mean=self.linear_systematics_mean, cov=self.linear_systematics_cov
        )
        return (
            normal.logpdf(model)
            + 0.5 * np.log(np.linalg.det(2 * np.pi * sig_linear))
            + prior.logpdf(mu_linear)
        )
