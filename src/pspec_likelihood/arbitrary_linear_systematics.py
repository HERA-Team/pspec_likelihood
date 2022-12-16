"""Module responsible for computing the likelihood for linear systematics."""
from __future__ import annotations

import warnings

import attr
import numpy as np
from astropy import units as un
from cached_property import cached_property
from numpy import matmul as matrix_multiply
from numpy.linalg import inv as inverse
from scipy import stats

from . import types as tp
from .likelihood import PSpecLikelihood


@attr.s
class LikelihoodLinearSystematic(PSpecLikelihood):
    """Likelihood for the case of arbitrary linear systematic parameters.

    This code assumes that the systematic parameters are linear and enter the
    likelihood through:

    .. math:: r′(θNL, θLsys) = d − 𝑚(θ_NL) − a_basis*θ_linearsys

    where d is the data vector, m(θ_NL) is the theory function and
    a_basis*θ_linearsys are the systematic parameters.
    The linear systematic parameters are then marginalized over.
    The likelihood in this code follows that of Tauscher et al. (2021).

    Parameters
    ----------
    linear_systematics_basis_function
        Callable function used to compute the linear basis of the linear systematics.
        This function must take the following parameters as input
        ``linear_systematic_basis_function(theta_lin, theta_nonlin, kperp_bins_theory)``
        The output of this function is required to be an ndarray of shape
        ``sys_params['linear'].shape``.
    mu_theta
        The mean of the Gaussian prior on the linear systematic variables.
        Must be a 1D array of length len(linear_systematics). Defaults to None.
        If None, the prior on the linear systematics are assumed to be improper uniform.
    sigma_theta
        The prior covariance of the linear systematic parameters in the case of a
        Gaussian prior on the linear systematic parameters. Must be an array of shape
        ``len(linear systematics) X len(linear_systematics)``.
        If None, the prior on the linear systematics is assumed to improper uniform.
    nlinear
        The number of linear parameters -- not required if ``mu_theta`` or
        ``sigma_theta`` are given.
    cov_tolerance
        Tolerance on the eigenvalues of the covariance to determine if covariance is
        singular.
    """

    linear_systematics_basis_function = attr.ib()
    mu_theta: np.ndarray | None = attr.ib(None)
    nlinear: int = attr.ib()
    sigma_theta: np.ndarray = attr.ib()
    cov_tolerance: float = attr.ib(default=10, converter=float)

    @cached_property
    def covariance_inv(self):
        """Computes inverse of covariance."""
        return inverse(self.model.covariance)

    @cached_property
    def is_improper_uniform(self) -> bool:
        """Checks for improper uniform priors."""
        return self.mu_theta is None

    @nlinear.default
    def _nlinear(self) -> int:
        if self.is_improper_uniform:
            raise ValueError("You need to provide nlinear if mu_theta is not provided")
        return len(self.mu_theta)

    @sigma_theta.default
    def _sigma_theta_default(self) -> np.ndarray:
        if self.is_improper_uniform:
            return np.diag(np.ones(self.nlinear) * np.inf)
        else:
            raise ValueError("Provide sigma_theta")

    @sigma_theta.validator
    def _sigma_theta_validator(self, att, val):
        if val.ndim != 2:
            raise ValueError("Covariance must be two dimensional")
        if val.shape[0] != val.shape[1]:
            raise ValueError("Covariance must be square")
        lambdas = np.linalg.eigvalsh(val)
        if not np.all(lambdas > -self.cov_tolerance):
            raise ValueError("Covariance is not invertible")

    @cached_property
    def sigma_theta_inv(self) -> np.ndarray:
        """Computes the inverse of the theta covariances."""
        return inverse(self.sigma_theta)

    @cached_property
    def compute_mn_prior(self):
        """Computes the multivariate normal prior."""
        return stats.multivariate_normal(mean=self.mu_theta, cov=self.sigma_theta)

    @cached_property
    def compute_mn_loglike(self):
        """Computes the multivariate normal for loglike."""
        return stats.multivariate_normal(cov=self.model.covariance)

    def compute_sigma_linear(self, basis: tp.PowerType) -> np.ndarray:
        """Computes sigma_linear given a basis."""
        a_sigma_a = matrix_multiply(
            basis.T, matrix_multiply(self.covariance_inv, basis)
        ).to_value("")

        return inverse(self.sigma_theta_inv + a_sigma_a)

    def compute_mu_linear(
        self, basis: tp.PowerType, sigma_linear: np.ndarray, resid: tp.PowerType
    ) -> np.ndarray:
        """Computes mu_linear given a basis and sigma_linear."""
        a_sigma_r = matrix_multiply(
            basis.T, matrix_multiply(self.covariance_inv, resid)
        )
        if not self.is_improper_uniform:
            a_sigma_r += matrix_multiply(self.sigma_theta_inv, self.mu_theta)

        return matrix_multiply(sigma_linear, a_sigma_r)

    def loglike(self, theory_params, sys_params):
        """
        Likelihood function for the arbitrary lineary systematics model.

        Details on this derivation can be found in
        Tauscher et al (2021).
        """
        # this is the linear systematic coefficients - compute at each step
        a_basis = self.linear_systematics_basis_function(
            sys_params, self.model.kperp_bins_obs, self.model.kpar_bins_obs
        )

        if not isinstance(a_basis, un.Quantity) or not a_basis.unit.is_equivalent(
            un.mK**2
        ):
            raise ValueError(
                "The function linear_systematics_basis_function must return a "
                "power-like quantity (in equivalently mK^2)"
            )

        # The shape of a_basis must match the k bins
        if a_basis.shape != (len(self.model.kpar_bins_obs), self.nlinear):
            raise ValueError(
                "linear_systematics_basis_function must return a (kbins, nlinear) array"
            )

        # compute the theoretical model for this set of parameters
        theory_model = self.model.compute_model(theory_params, sys_params)

        r = self.power_spectrum - theory_model

        # compute the values of sigma_linear for this basis
        sigma_linear = self.compute_sigma_linear(a_basis)

        # compute the values of mu_linear for this basis and sigma_linear
        mu_linear = self.compute_mu_linear(a_basis, sigma_linear, r)

        rprime = r - np.matmul(a_basis, mu_linear)

        if not self.is_improper_uniform:
            prior = np.sum(self.compute_mn_prior.logpdf(mu_linear))
        else:
            prior = 0

        loglikelihood = np.sum(self.compute_mn_loglike.logpdf(rprime))

        loglikelihood_eff = (
            0.5 * np.linalg.slogdet(sigma_linear)[1] + prior + loglikelihood
        )

        if not np.isfinite(loglikelihood_eff):
            warnings.warn("Non-finite likelihood")

        return loglikelihood_eff
