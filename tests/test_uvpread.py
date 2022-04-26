"""Test loading an UVPspec file"""
import astropy.units as un

dummy_theory_model = lambda z,k: 1*un.mK**2
dummy_sys_model = lambda z,k: 1*un.mK**2

def test_uvpread():
    from pspec_likelihood import DataModelInterface
    import numpy as np
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5")
    dmi1 = DataModelInterface.from_uvpspec(uvp1, band_index=1,
        theory_model=dummy_theory_model, sys_model=dummy_sys_model,
        kpar_bins_theory = np.ones(40), kperp_bins_theory=None,
        kpar_widths_theory=np.ones(40), kperp_widths_theory=np.ones(40))
    assert dmi1.covariance is not None # there is data
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
    uvp = ds.pspec(baselines1, baselines2, dsets=(0, 1),
                   pols=[('xx', 'xx')], spw_ranges=(175,195),
                   taper='bh', store_cov=True, verbose=False)
    spw = 0
    print(uvp.cov_array_real[spw].shape)
    print(uvp.Nblpairts, uvp.Ndlys, uvp.Ndlys, uvp.Npols)
    # From https://github.com/HERA-Team/hera_pspec/tree/main/hera_pspec/data
    uvp2 = uvp#DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/test_uvp.h5")
    uvp2.cosmo = uvp1.cosmo
    dmi2 = DataModelInterface.from_uvpspec(uvp2, band_index=0,
        theory_model=dummy_theory_model, sys_model=dummy_sys_model,
        kpar_bins_theory = np.ones(20), kperp_bins_theory=np.ones(12),
        kpar_widths_theory=np.ones(20), kperp_widths_theory=np.ones(12))
    assert dmi2.kperp_bins_obs is not None # data should _not_ be sperically averaged
    