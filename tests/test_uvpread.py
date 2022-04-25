"""Test loading an UVPspec file"""

def test_uvpread():
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    from pspec_likelihood import DataModelInterface
    uvp1 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5")
    dmi1 = DataModelInterface.from_uvpspec(uvp1, band_index=1)
    assert dmi1.covariance is not None # there is data
    assert dmi1.kperp_bins_obs is None # data should be sperically averaged
    # Note: Spherical average message usually in `uvp.history.split("\n")[0]`

    uvp2 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/test_uvp.h5")
    dmi2 = DataModelInterface.from_uvpspec(uvp2, band_index=0)
    assert dmi2.kperp_bins_obs is not None # data should _not_ be sperically averaged
