"""Test loading an UVPspec file"""
import os

import astropy.units as un
import hera_pspec as hp
import numpy as np
import pytest
from astropy.cosmology import units as cu
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData

from pspec_likelihood import DataModelInterface


def dummy_theory_model(z, k):
    return 1 * un.mK**2


def dummy_sys_model(z, k):
    return 1 * un.mK**2


def prepare_uvp_object(
    path_to_wf="tests/data/",
    dfile="data_calibrated_testfile.h5",
    time_avg=True,
    spherical_avg=True,
    redundant_avg=True,
):
    # Based on https://github.com/HERA-Team/pspec_likelihood/blob/api_idr2like/
    # dvlpt/tests_data_file.ipynb
    uvd = UVData()
    uvd.read_uvh5(os.path.join(path_to_wf, dfile))
    # beam
    beamfile = os.path.join(DATA_PATH, "HERA_NF_pstokes_power.beamfits")
    uvb = hp.pspecbeam.PSpecBeamUV(beamfile)
    # Create a new PSpecData object, and don't forget to feed the beam object
    ds = hp.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)
    ds.Jy_to_mK()
    # choose baselines
    baselines1, baselines2, blpairs = hp.utils.construct_blpairs(
        uvd.get_antpairs(), exclude_permutations=True, exclude_auto_bls=True
    )
    # compute ps
    uvp = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("pI", "pI")],
        spw_ranges=(175, 195),
        taper="bh",
        verbose=False,
        store_cov=True,
        store_window=True,
        baseline_tol=100.0,
    )
    print(
        "There are",
        uvp.Nblpairs,
        "baseline pairs at",
        uvp.Ntimes,
        "times -->",
        uvp.data_array[0].shape[0],
    )
    print("Furthermore we have", uvp.data_array[0].shape[1], "kparas (freq).")
    print("Thus the data array has shape", uvp.data_array[0].shape)
    # get redundant groups
    blpair_groups, blpair_lens, _ = uvp.get_red_blpairs()
    # there are baseline pairs that do not belong to a redundant group
    extra_blpairs = set(uvp.blpair_array) - {
        blp for blpg in blpair_groups for blp in blpg
    }
    # only keep blpairs in redundant groups
    uvp.select(blpairs=[blp for blpg in blpair_groups for blp in blpg])
    print(
        "Out of those baselines pairs skip",
        len(extra_blpairs),
        "that are not part of redunant groups, leaving",
        uvp.Nblpairs,
        "in two groups of",
        len(blpair_groups),
        ":",
        blpair_groups,
    )
    print("Data array shape with these only", uvp.data_array[0].shape)
    # perform redundant average
    if redundant_avg:
        uvp.average_spectra(blpair_groups=blpair_groups)
    print("Data array shape after redundant average", uvp.data_array[0].shape)
    if time_avg:
        uvp.average_spectra(time_avg=time_avg)
        print("Data array shape after time average", uvp.data_array[0].shape)
    kbins = np.linspace(0.1, 2.5, 40)
    if spherical_avg:
        sph = hp.grouping.spherical_average(uvp, kbins, np.diff(kbins).mean())
        print("Data array shape after spherical average", sph.data_array[0].shape)
        return sph
    else:
        return uvp


def test_spherical_ps():
    uvp = prepare_uvp_object()
    dmi = DataModelInterface.from_uvpspec(
        uvp=uvp, spw=0, theory_model=dummy_theory_model, theory_uses_spherical_k=True
    )
    assert np.shape(dmi.covariance) == (40, 40)  # right shape
    assert np.shape(dmi.kpar_bins_obs) == (40,)  # right shape
    assert dmi.kperp_bins_obs is None  # data should be sperically averaged
    return dmi


def test_cylindrical_ps():
    uvp = prepare_uvp_object(spherical_avg=False)
    dmi = DataModelInterface.from_uvpspec(
        uvp=uvp, spw=0, theory_model=dummy_theory_model, theory_uses_spherical_k=True
    )
    assert np.shape(dmi.covariance) == (40, 40)  # right shape
    assert np.shape(dmi.kpar_bins_obs) == (40,)  # right shape
    assert np.shape(dmi.kperp_bins_obs) == (40,)  # right shape
    return dmi


def test_exception_no_time_avg():
    uvp = prepare_uvp_object(time_avg=False)
    with pytest.raises(ValueError) as e:
        DataModelInterface.from_uvpspec(
            uvp=uvp,
            spw=0,
            theory_model=dummy_theory_model,
            theory_uses_spherical_k=True,
        )
        print(e)
        assert str(e.value) == (
            "The UVPSpec object has not been fully time-"
            "averaged. Please time-average with uvp."
            "average_spectra(time_avg=True) before passing to "
            "DataModelInterface.from_uvpspec"
        )


def test_warning_not_redundantly_averaged():
    uvp = prepare_uvp_object(redundant_avg=False, spherical_avg=False)
    with pytest.warns(
        UserWarning, match="The UVPSpec object is not redundantly averaged."
    ):
        DataModelInterface.from_uvpspec(
            uvp=uvp,
            spw=0,
            theory_model=dummy_theory_model,
            theory_uses_spherical_k=True,
        )


def test_exception_no_units():
    dfile = "zen.2458116.31939.HH.uvh5"
    datafile = os.path.join(DATA_PATH, dfile)
    uvd = UVData()
    uvd.read_uvh5(datafile)
    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    uvb = hp.pspecbeam.PSpecBeamUV(beamfile, cosmo=None)
    jy_to_mk = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol="xx")
    # reshape to appropriately match a UVData.data_array object and multiply in!
    uvd.data_array *= jy_to_mk[None, None, :, None]
    # Create a new PSpecData object, and don't forget to feed the beam object
    ds = hp.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)
    # choose baselines
    baselines1, baselines2, blpairs = hp.utils.construct_blpairs(
        uvd.get_antpairs()[1:], exclude_permutations=False, exclude_auto_bls=True
    )
    # compute ps
    uvp = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=(175, 195),
        taper="bh",
        store_cov=True,
        verbose=False,
    )
    with pytest.raises(ValueError, match=r"Power Spectrum must be in"):
        with pytest.warns(UserWarning, match="Converting to Delta^2 in place..."):
            DataModelInterface.from_uvpspec(
                uvp,
                spw=0,
                theory_model=dummy_theory_model,
                sys_model=dummy_sys_model,
                kpar_bins_theory=np.ones(20 * 12) * (cu.littleh / un.Mpc),
                kperp_bins_theory=np.ones(20 * 12) * (cu.littleh / un.Mpc),
                kpar_widths_theory=np.ones(20 * 12) * (cu.littleh / un.Mpc),
                kperp_widths_theory=np.ones(20 * 12) * (cu.littleh / un.Mpc),
            )


def test_IDR2_file(uvp1):  # noqa: N802
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    dmi1 = DataModelInterface.from_uvpspec(
        uvp1,
        spw=1,
        theory_model=dummy_theory_model,
        sys_model=dummy_sys_model,
        kpar_bins_theory=np.ones(40) * (1 / un.Mpc),
        kperp_bins_theory=None,
        kpar_widths_theory=np.ones(40) * (1 / un.Mpc),
    )
    assert np.shape(dmi1.covariance) == (40, 40)  # right shape
    assert dmi1.kperp_bins_obs is None  # data should be sperically averaged
    # Note: Spherical average message usually in `uvp.history.split("\n")[0]`
