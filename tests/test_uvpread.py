"""Test loading an UVPspec file"""

def test_uvpread():
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    from pspec_likelihood import DataModelInterface
    uvp = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5")
    dmi = DataModelInterface.from_uvpspec(uvp, band_index=1)
    assert dmi.covariance is not None
