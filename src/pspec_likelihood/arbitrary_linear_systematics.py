"""Module responsible for computing the likelihood for linear systematics."""
import numpy as np
from numpy import matmul as matrix_multiply
from numpy.linalg import det as determinant
from numpy.linalg import inv as inverse


class LikelihoodLinearSystematic:
    """
    Likelihood for the case of arbitrary linear systematic parameters.

    This code assumes that the systematic parameters are linear and enter the
    likelihood through :
    r′(θNL, θLsys) = d − 𝑚(θ_NL) − a_basis*θ_linearsys ≡ r − a_basis*θ_linearsys,
    where d is the data vector, m(θ_NL) is the theory function and
    a_basis*θ_linearsys are the systematic parameters.
    The linear systematic parameters are then marginalized over.
    The likelihood in this code follows that of Tauscher et al. (2021).

    Input parameters
    ----------------
    sys_params
    This is a dictionary of the theory and systematic parameters.
    Keys are as follows:

    'non-linear': these are the input parameters for the theoretical_model.
        These parameters are assumed to be an ndarray

    'linear' : these are the linear systematics which are marginalizaed.
        These parameters are assumed to be an ndarray

     'linear_systematics_basis_function': callable function used to compute
        the linear basis of the linear systematics.
        This function must take the following parameters as input
        linear_systematic_basis_function(theta_linear, theta_non_linear,
        model.kperp_bins_theory)
        The output of this function is required to be an ndarray of
        shape sys_params['linear'].shape

    'covariance' : a covariance matrix, 𝚺 of shape sys_params['linear'].shape,
        that encodes the Gaussian thermal-noise on each of
        the power spectrum measurements

     'improper_uniform' : bool. If true, the prior on the linear systematic priors
        is taking to be improper uniform. If False, the prior on the linear
        systematic parameters is assumed to be Gaussian. In this case an
        additional covariance and mean, which characterize the Gaussian prior
        on theta_linear are required to be supplied in order to marginalize
        over the Gaussian prior.

     data : the dataset that we are using in the inference. This is an
     array of shape

     tolerance : limit to which we are willing to tolerate negative eigenvalues
     of a covariance matrix. Default 1e-8. Negative eigenvalues of the covariance
     matrix lead to non-finite values of the likelihood

     sigma_theta : the covariance of the linear systematic parameters in the case
        of a Gaussian prior on the linear systematic parameters. Must be an array of
        shape len(linear systematics) X len(linear_systematics). Defaults to None.
        If None, the prior on the linear systematics is assumed to improper uniform.

     mu_theta : the mean of the Gaussian prior on the linear systematic variables.
        Must be a 1D array of length len(linear_systematics). Defaults to None.
        If None, the prior on the linear systematics are assumed to be improper uniform.

     Returns
     -------
     likelihood: (float)

    """

    def __init__(
        self,
        data,
        linear_systematics_basis_function=linear_systematics_basis_function,
        covariance=covariance,
        mu_theta=None,
        sigma_theta=None,
        tolerance=1e-8,
        model=data_model,
    ):
        def cov_check(cov, tol=tolerance):
            lambdas = np.linalg.eigvalsh(cov)
            return np.all(lambdas > -tol)

        # make the DataModelInterface object an explicit object of this class
        self.model = model

        # data vector
        self.data_vector = self.model.data

        # model to compute the theory function
        self.compute_theory_model = self.model.compute_model

        """
        The kperp bins from data model interface.
        Used to comptue the linear basis function
        """
        self.kperp_bins = self.model.kperp_bins_theory

        # function used to compute the linear systematics
        self.linear_systematic_basis_function = linear_systematic_basis_function

        # mean on the Gaussian prior on the linear systematic parameters
        self.mu_theta = mu_theta

        # covariance of the Gaussian prior on the linear systematic parameters
        self.sigma_theta = sigma_theta

        """
        covariance matrix which encodes gaussian thermal
        noise of power spectrum measurements
        """
        self.sigma = covariance

        """ Check whether the covariance is semi positive definite."""
        if not cov_check(self.sigma):
            raise Exception("covariance is not invertible")
        else:
            self.sigma_inv = inverse(self.sigma)

        """
        If the prior is improper uniform, then we don't need to specify
        anything about the parameter space of the systematic priors.
        """
        if self.sigma_theta is None or self.mu_theta is None:
            r"""
            this is a hacky way of getting the prior on the
            linear systematics to match the improper uniform result
            i.e. \pi_L( \mu_L ) --> 1
            """
            self.sigma_theta_inv = 0
            self.sigma_theta = np.identity(len(self.mu_theta)) * np.pi ** (
                1 / len(self.mu_theta)
            )
        else:
            """
            The user needs to specify the mean and covariance of the Gaussian
            which characterizes the prior of the linear systematics
            make sure that the sigma_theta and mu_linear are
            arrays of the correct shape.
            """
            if not isinstance(self.sigma_theta, np.ndarray):
                print("theta covariance is of type ", type(self.sigma_theta))
            if not isinstance(self.mu_theta, np.ndarray):
                print("theta covariance is of type ", type(self.mu_theta))

            # The shape of mu_theta must match the covariance
            assert (
                len(self.mu_theta) == self.sigma_theta.shape[0]
                and len(self.mu_linear) == self.sigma_theta.shape[1]
            ), f'{"Dimension the systematics does not match systematic params"}'

            # make sure the covariance is invertible
            if not cov_check(self.sigma_theta):
                raise Exception("covariance is not invertible")

            self.sigma_theta_inv = inverse(self.sigma_theta)

    def loglike(self, theory_params, sys_params):
        """
        Linear systematic parameters.

        Right now the systematic variables are ASSUMED to
        be in the same basis as the k_bins.
        """
        theta_linear = sys_params["linear"]

        # non-linear systematic parameters
        theta_nl = sys_params["non_linear"]

        # this is the linear systematic coefficients - compute at each step
        a_basis = self.linear_systematic_basis_function(
            theta_linear, theta_nl, self.kperp_bins
        )

        # optionally apply window function to the linear systematics matrix here

        # compute the theoretical model for this set of parameters
        theory_model = self.compute_theory_model(theta_nl, theta_linear, a_basis)

        # The shape of a_basis must match the k bins
        assert a_basis.shape[0] == len(theta_nl) and self.a_basis.shape[1] == len(
            theta_nl
        ), f'{"Dimensionality of systematics does not match systematic params"}'

        # make sure the covariance is of the same shape as the linear systematics
        assert (
            a_basis.shape == self.sigma.shape
        ), f'{"Dimensionality of the systematics does not match covariance"}'

        r = self.data_vector - theory_model
        rprime = r - np.matmul(a_basis, theta_linear)

        """
        Details on this derivation can be found in
        Tauscher et al (2021).
        """

        def compute_sigma_linear(self, basis):
            """Computes sigma_linear given a basis."""
            sigma_linear = inverse(
                self.sigma_theta_inv
                + matrix_multiply(basis.T, matrix_multiply(self.sigma_inv, basis))
            )
            return sigma_linear

        def compute_mu_linear(self, basis, sigma_linear, r):
            """Computes mu_linear given a basis and sigma_linear."""
            coefficient = matrix_multiply(
                self.sigma_theta_inv, self.mu_theta
            ) + matrix_multiply(basis.T, matrix_multiply(self.sigma_inv, r))
            return matrix_multiply(sigma_linear, coefficient)

        def compute_h(self, r, basis):
            """
            Little h.

            Computes little h from Tauscher et al (2021).
            No relation to cosmic little h.
            """
            return matrix_multiply(
                self.sigma_theta_inv, self.mu_theta
            ) + matrix_multiply(basis.T, matrix_multiply(self.sigma_inv, r))

        def compute_b(self, h):
            """Computes little b from Tauscher et al (2021)."""
            return -matrix_multiply(h.T, matrix_multiply(self.sigma_linear, h))

        # compute the values of sigma_linear for this basis
        sigma_linear = self.compute_sigma_linear(a_basis)

        # compute the values of mu_linear for this basis and sigma_linear
        mu_linear = self.compute_mu_linear(a_basis, sigma_linear, r)

        """
        computing b and h in Tauscher et al is not used in this implementation
        b = self.compute_b(h)
        h = self.compute_h(r, a_basis)
        """

        """ the effective likelihood (i.e. the likelihood
        marginalized over the linear systematic parameters) is
        the product of the prior which depends only on the
        linear systematics and a likelihood
        which takes into account the non-linearity"""

        r_prior = mu_linear - self.mu_theta

        prior = np.pi * determinant(self.sigma_theta) - 0.5 * matrix_multiply(
            r_prior.T, matrix_multiply(self.sigma_theta_inv, r_prior)
        )

        loglikelihood_nl = np.pi * determinant(self.sigma) - 0.5 * matrix_multiply(
            rprime.T, matrix_multiply(self.sigma_inv, rprime)
        )

        loglikelihood_eff = np.pi * sigma_linear + prior + loglikelihood_nl

        if np.isfinite(loglikelihood_eff) is False:
            print("Warning: Non definite likelihood")

        return loglikelihood_eff
