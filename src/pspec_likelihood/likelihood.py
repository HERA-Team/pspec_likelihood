"""Primary module defining container for power spectrum measurements and models."""
import attr
import numpy as np
from cached_property import cached_property
from hera_pspec import grouping
from hera_pspec.uvpspec import UVPSpec
from scipy.integrate import quad

from .utils import listify


@attr.s(frozen=True)
class PSpecLikelihood:
    r"""Container for power spectrum measurements and models.

    This class keeps track of power-spectrum measurements
    (and their associated covariances and window functions)
    along with a theoretical model and calculations of the likelihoods
    given this model that propertly account for the window functions.
    For now, this container assumes Gaussian measurement errors and
    thus only keeps track of covariances but this may change in the future.

    Parameters
    ----------
    ps_files : list of str or Path
        List of uvpspec files that constitute power-spectrum measurements,
        or a single filename. The current framework assumes that
        each power spectrum measurement (or spectral window) is statistically
        independent.
    theoretical_model : func(k, z, little_h, **params) -> delta_sq [mK^2]
        a function that takes as its arguments a numpy vector of k-values (floats),
        a bool (little_h), and any number of additional parameters and returns a vector
        of floats with the same shape as the k-vector. little_h specifies whether k
        units are in h/Mpc or 1/Mpc.
    bias_model : func(k, z, little_h, **params) -> delta_sq [mK^2]
        a function that takes in as its arguments a numpy vector of k-values (floats)
        and a bool (little_h) and any additional number of theory params and returns
        a vector of floats with the same shape as the k-vector.
        The nuisance model is defined in data space
        and can be treaded as the bias term in
        \hat{p} = W p_true + b
    little_h
        specifies whether k units are in h/Mpc or 1/Mpc
    bias_prior : func(params) -> prob
        a function that takes as its arguments a dictionary of nuisance parameters
        and returns a prior probability for these parameters.
    k_bins : array-like floats
        a list of floats specifying the centers of k-bins.
    history : str
        string with file history.
    param_names: list of strings
        list of parameter names if params is list, otherwise None.
        log_unnormalized_likelihood can take params as either a dictionary
        or a list of values. In the latter case, param_names needs to
        be given as the list of corresponding parameter names (keys) so
        the list can internally be converted to a dictionary.
    """

    ps_files = attr.ib(converter=listify)
    theoretical_model = attr.ib(validator=attr.validators.is_callable())
    bias_model = attr.ib(validator=attr.validators.is_callable())
    k_bin_widths = attr.ib(converter=np.ndarray)
    k_bin_centres = attr.ib(converter=np.ndarray)

    little_h = attr.ib(
        True, converter=bool, validator=attr.validators.instance_of(bool)
    )
    weight_by_cov = attr.ib(
        True, converter=bool, validator=attr.validators.instance_of(bool)
    )
    history = attr.ib("", converter=str)
    run_check = attr.ib(True, converter=bool)
    param_names = attr.ib(None, converter=attr.converters.optional(tuple))

    @ps_files.validator
    def _check_existence(self, att, val):
        for fl in val:
            if not fl.exists():
                raise FileNotFoundError(f"{fl} doesn't exist")

    @k_bin_centres.validator
    @k_bin_widths.validator
    def _check_kbins(self, att, val):
        if not np.isrealobj(val):
            raise TypeError("k_bins must be real numbers")

        if not len(val):
            raise ValueError("k_bins must have at least one element.")

    @cached_property
    def measurements(self):
        """The UVPSpec measurements."""
        uvp_in = UVPSpec(self.ps_files)
        return grouping.spherical_average(
            uvp_in,
            self.k_bin_centers,
            self.k_bin_widths,
            time_average=True,
            weight_by_cov=self.weight_by_cov,
            add_to_history="spherical average with time averaging.",
            little_h=self.little_h,
            run_check=self.run_check,
        )

    def discretized_ps(self, spw, theory_params, little_h=True, method=None):
        r"""Compute the power spectrum in the specified spectral windows and k_bins.

        Our analysis formalism assumes that the power spectrum is piecewise
        constant (see e.g. arXiv 1103.0281, 1502.0616). Therefore we bin the
        power spectrum to the bins given as properties of the class. The
        redshifts are determined by the spherical windows (spw).

        Possible methods: Just evaluate the power spectrum at the bin centers,
        integrate over the power spectrum to take the bin average, or
        evaluate at the bin edges (+ center?) and return the mean.

        Parameters
        ----------
        spw
            spherical windows
        theory_params
            dictionary containing parameters for the theory model
        little_h
            bool specifying whether k units are in h/Mpc or 1/Mpc
        method
            One of 'bin_center', 'two_point' or 'integrate'. The method of defining
            value within a bin.

        Returns
        -------
        results
            list of power spectrum values corresponding to the bins
        errors
            Estimation of the error through binning, if a suitable method has been
            chosen, otherwise None.
        """
        z = self.get_z_from_spw(spw)

        # Q: Is little_h a keyword argument of theory_func?
        # Q: Does the bin go from center-width/2 to center+width/2 ?
        # The error is just an order of magnitude, not any precise confidence interval.
        # If the power spectrum was monotonous, the error would be the maximal
        # deviation.

        if method == "bin_center":
            results = self.theoretical_model(
                self.k_bin_centers, z, little_h, theory_params
            )
            errors = None
        elif method == "two_point":
            lower = self.theoretical_model(
                self.k_bin_centers - self.k_bin_widths / 2, z, little_h, theory_params
            )
            upper = self.theoretical_model(
                self.k_bin_centers + self.k_bin_widths / 2, z, little_h, theory_params
            )
            results = (lower + upper) / 2
            errors = (lower - upper) / 2
        elif method == "integrate":

            def pk_func(k):
                return self.theoretical_model(k, z, little_h, theory_params)

            results = []
            errors = []
            for center, width in zip(self.kbin_centers, self.kbin_widths):
                result, error = quad(pk_func, center - width / 2, center + width / 2)
                results.append(result / width)
                errors.append(error / width)
        else:
            raise ValueError(
                f"method must be one of 'bin_center', 'two_point' or 'integrate'. "
                f"Got '{method}'."
            )

        return results, errors

    def windowed_theoretical_ps(self, spw, theory_params):
        r"""Calculate theoretical power spectrum with data window function applied.

        Also apply appropriate frequency / k-averaging/binning to theoretical model.

        Parameters
        ----------
        theory_params : dict
            dictionary of theoretical parameters.
        spectral_window : int
            number of spectral window to generate windowed theoretical ps.
        little_h : bool, optional
            if true, use little_h units (e.g. h^-1 Mpc)

        Returns
        -------
        A vector of floats, p_w = W p_m
        where p_m is a theoretical model power spectrum, W is the window function
        applied to that model.
        """
        # Need to specify appropriate k-averaging.
        # Below, we just have sampling.
        discretized_ps = self.discretized_ps(spw, theory_params)
        windows_ps = self.measurements.get_window_function(spw)
        return discretized_ps, windows_ps

    def get_z_from_spw(self, spw):
        r"""Get redshift from a spectral window."""
        # TODO: get redshift(s) z from spw / integrate
        raise NotImplementedError("Need to implement this.")

    def params_to_dict(self, params):
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
            if self.param_names is not None:
                assert set(self.param_names) == set(
                    params.keys()
                ), "input parameters don't match parameters of the likelihood"
            return params
        else:
            if self.param_names is None:
                raise ValueError(
                    "To pass params as a sequence, rather than dict, likelihood must be"
                    " created with param_names"
                )
            assert isinstance(
                params, [list, tuple]
            ), "input parameters must be either dict, list or tuple"
            return dict(zip(self.param_names, params))

    def log_unnormalized_likelihood(self, params):
        r"""
        log-likelihood for set of theoretical and bias parameters.

        Probability of data given a model (this is distinct from a properly normalized
        posterior).

        Parameters
        ----------
        params : dictionary or list
            theoretical and systematics parameters to compute likelihood for.
            This is the only function that accepts params as list or dict, other
            functions get called from here and take a dictionary.
        """
        params = self.params_to_dict(params)
        raise NotImplementedError

    def __call__(self, *params):
        """Get the log-likelihood for a given set of parameters."""
        pass
