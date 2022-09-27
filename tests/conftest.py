from pathlib import Path

import hera_pspec as hp
import pytest

DATA_PATH = Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def uvp1():
    uvp = hp.UVPSpec()
    uvp.read_hdf5(DATA_PATH / "pspec_h1c_idr2_field1.h5")
    return uvp
