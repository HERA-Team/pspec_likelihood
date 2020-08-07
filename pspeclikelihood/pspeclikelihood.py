from hera_pspec import uvpspec
from hera_pspec import grouping
import numpy as np

class PSpecLikelihood():
    """Container for power spectrum measurements and models.
    
    
    Functionality that we'd want for an analysis:
        1) comparing theory / systematics to data on equal footing.
        2) calculating likelihoods.
            This is the ultimate goal of this class.
    --------------------------------------
        3) sampling the posterior.
            Sampler goes outside of this particular class.
        4) confidence intervals for astrophysics/cosmology.
            Also outside of this class.
        
    Where do all of these belong? 
    
        
        
    This class keeps track of power-spectrum measurements
     (and their associated covariances and window functions)
     along with a theoretical model and calculations of the likelihoods
     given this model that propertly account for the window functions.
     For now, this container assumes Gaussian measurement errors and
     thus only keeps track of covariances but this may change in the future.

     This class also only considers additive nuisance models.
     
     How do we keep track of sampling?
    """

  def __init__(self, ps_files, theoretical_model, bias_model,
               bias_prior, kbin_centers, kbin_widths, little_h=True,
               weight_by_cov=True, history='', run_check=True):
      """Initializes PspecModel object.

      Parameters
      ----------
      ps_files : list of uvpspec files
        List of uvpspec files that constitute our power-spectrum measurements
        or a single uvpspec object. Our current framework assumes that these
        each power spectrum measurement (or spectral window) are statistically
        independent.

      theoretical_model : func(k, z, little_h, **params) -> delta_sq [mK^2]
        a function that takes as its arguments a numpy vector of k-values (floats)
        ,a bool (little_h), and any number of additional parameters and returns a vector of
        floats with the same shape as the k-vector. little_h specifies whether k
        units are in h/Mpc or 1/Mpc.

      theoretical_prior : func(params) -> prob
        a function that takes as its arguments a dictionary of theoretical parameters
        and returns a prior probability for these parameters.

      bias_model : func(k, z, little_h, **params) -> delta_sq [mK^2]
        a function that takes in as its arguments a numpy vector of k-values (floats)
        and a bool (little_h) and any additional number of theory params and returns
        a vector of floats with the same shape as the k-vector.
        The nuisance model is defined in data space
        and can be treaded as the bias term in
        \hat{p} = W p_true + b
        little_h specifies whether k units are in h/Mpc or 1/Mpc

     bias_prior : func(params) -> prob
        a function that takes as its arguments a dictionary of nuisance parameters
        and returns a prior probability for these parameters.

     k_bins : array-like floats
        a list of floats specifying the centers of k-bins.
        
     history : str
        string with file history.

      Attributes
      -------
        measurements : UVPspec
            uvpspec object generated from list of uvpspec files. Each
            spectral window is assumed to be statistically independent.

        theory_func : func(k, params, little_h, ps_units) -> p(k), C(k, k')
            function provided in theoretical_model arg.

        theoretical_prior : func(params) -> prob
            function provided in theoretical_prior
            
        bias_model : func(k, params, little_h, ps_units) -> p(k), C(k, k')
            function provided in nuisance_model

        bias_prior : func(params) -> prob
            functin provided in nuisance_prior

        history : str
            string provided in history arg.
      """
      if isinstance(ps_files, string):
          ps_files = [ps_files]
      # what assumptions are we making about the power spectra?
      # are they spherically averaged?
      if not isinstance(ps_files, list):
          raise ValueError("ps_files must be a string or a list of strings.")
      uvp_in = UVPspec(ps_file_list)
      # spherically average all power spectrum measurements using inverse covariance
      # weighting.
      self.measurements = grouping.spherical_average(uvp_in, kbin_centers, kbin_widths,
                                                     time_average=True, weight_by_cov=weight_by_cov,
                                                     add_to_history='spherical average with time averaging.',
                                                     little_h=little_h, run_check=run_check)
      self.theoretical_model = theoretical_model
      self.nuisance_model = bias_model
      self.nuisance_model = bias_prior
      self.history = history
      self.little_h = little_h
      self.kbin_centers = kbin_centers
      self.kbin_widths = kbin_widths
      # add parameters that directly reference mean and covariance of measurements.
      # also add keywords that describe the data distribution.

  def windowed_theoretical_ps(spw, theory_params):
      """Calculate theoretical power spectrum with data window function applied.
         Also apply appropriate frequency / k-averaging/binning to theoretical model.
          
      """

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
        where p_m is a theoretical model power spectrum, W is the window function applied
        to that model.
      """
      # Need to specify appropriate k-averaging.
      # Below, we just have sampling.
      discretized_ps = self.discretized_ps(spw, theory_params)
      windows_ps = self.measurements.get_window_function(spw) @ discretized
      
      
 def log_unnormalized_likelihood(params):
    """
        log-likelihood for set of theoretical and bias parameters.
        Probability of data given a model (this is distinct from a properly normalized posterior).
        
        Parameters
        ----------
        
        params : arbitrary arguments
            theoretical and systematics parameters to compute likelihood for.
    """
    
