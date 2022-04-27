"""Test loading an UVPspec file"""
import astropy.units as un

dummy_theory_model = lambda z,k: 1*un.mK**2
dummy_sys_model = lambda z,k: 1*un.mK**2

def test_uvpread_averaged():
    from pspec_likelihood import DataModelInterface
    import numpy as np
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5")
    dmi1 = DataModelInterface.from_uvpspec(uvp1, band_index=1,
        theory_model=dummy_theory_model, sys_model=dummy_sys_model,
        kpar_bins_theory = np.ones(40), kperp_bins_theory=None,
        kpar_widths_theory=np.ones(40), kperp_widths_theory=np.ones(40))
    assert np.shape(dmi1.covariance) == (40, 40) # right shape
    assert dmi1.kperp_bins_obs is None # data should be sperically averaged
    # Note: Spherical average message usually in `uvp.history.split("\n")[0]`

def test_uvpread_non_averaged():
    from pspec_likelihood import DataModelInterface
    uvp1 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5")
    # Make a non spherically-averaged file from the UVData() object
    import numpy as np
    import sys
    import os
    import hera_pspec as hp
    from hera_pspec.data import DATA_PATH
    from pyuvdata import UVBeam, UVData
    dfile = 'zen.2458116.31939.HH.uvh5'
    datafile = os.path.join(DATA_PATH, dfile)
    # read data file into UVData object
    uvd = UVData()
    uvd.read_uvh5(datafile)
    # Create a new PSpecData object, and don't forget to feed the beam object
    ds = hp.PSpecData(dsets=[uvd, uvd], wgts=[None, None])
    # choose baselines
    baselines1, baselines2, blpairs = hp.utils.construct_blpairs(
        uvd.get_antpairs()[1:],
        exclude_permutations=False,
        exclude_auto_bls=True)
    # compute ps
    uvp2 = ds.pspec(baselines1, baselines2, dsets=(0, 1),
                   pols=[('xx', 'xx')], spw_ranges=(175,195),
                   taper='bh', store_cov=True, verbose=False)
    uvp2.cosmo = uvp1.cosmo
    dmi2 = DataModelInterface.from_uvpspec(uvp2, band_index=0,
        theory_model=dummy_theory_model, sys_model=dummy_sys_model,
        kpar_bins_theory = np.ones(20*12), kperp_bins_theory=np.ones(20*12),
        kpar_widths_theory=np.ones(20*12), kperp_widths_theory=np.ones(20*12))
    assert np.shape(dmi2.kperp_bins_obs) == (20*12,), np.shape(dmi2.kperp_bins_obs)
    assert np.shape(dmi2.kpar_bins_obs) == (20*12,), np.shape(dmi2.kpar_bins_obs)
    assert np.shape(dmi2.power_spectrum) == (20*12,), np.shape(dmi2.power_spectrum)
    assert np.shape(dmi2.covariance) == (20*12, 20*12), np.shape(dmi2.covariance)
    