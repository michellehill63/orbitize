"""Tests for orbitize.precession."""

import os
import tempfile

import numpy as np
import pytest

from orbitize import precession as prec


def test_heintz_magnitude_and_sign():
    # alpha = 90, delta = 0: rate = n exactly. A pole-of-date PA at t = 2010
    # leads J2000, so reducing to J2000 subtracts 10 n.
    d = prec.pa_precession_correction(90.0, 0.0, 2010.0, 2000.0, warn=False)
    assert np.isclose(d, -10 * 0.00557)
    # antisymmetric in RA about 180 deg
    assert np.isclose(prec.pa_precession_correction(270.0, 0.0, 2010.0, 2000.0, warn=False), -d)
    # zero at alpha = 0 / 180
    assert np.isclose(prec.pa_precession_correction(0.0, 45.0, 2010.0, warn=False), 0.0)
    assert np.isclose(prec.pa_precession_correction(180.0, 45.0, 2010.0, warn=False), 0.0, atol=1e-15)
    # sec(delta) amplification: delta = 60 deg doubles it
    assert np.isclose(prec.pa_precession_correction(90.0, 60.0, 2010.0, warn=False), 2 * d)


def test_round_trip():
    pa = 123.4
    p1 = prec.precess_pa(pa, 50.0, 40.0, 1950.0, 2000.0)
    p2 = prec.precess_pa(p1, 50.0, 40.0, 2000.0, 1950.0)
    assert np.isclose(p2, pa)


@pytest.mark.parametrize(
    "ra,dec,ep",
    [(90, 30, 2010), (270, -30, 1990), (45, 60, 1900), (300, -70, 1850), (120, 10, 2025)],
)
def test_heintz_matches_rigorous(ra, dec, ep):
    pytest.importorskip("astropy")
    lin = prec.pa_precession_correction(ra, dec, ep, 2000.0, warn=False)
    rig = prec.pa_precession_correction_rigorous(ra, dec, ep, 2000.0)[0]
    # first-order formula is good to a few % of the correction (+1e-3 deg)
    # even over 150 yr away from the poles
    assert abs(lin - rig) < 0.05 * abs(rig) + 1e-3


def test_rotate_radec_consistent_with_seppa():
    from orbitize.system import radec2seppa, seppa2radec

    sep, pa, dth = 500.0, 37.0, 0.3
    ra, dec = seppa2radec(sep, pa)
    ra2, dec2 = prec.rotate_radec(ra, dec, dth)
    sep2, pa2 = radec2seppa(ra2, dec2)
    assert np.isclose(sep2, sep)
    assert np.isclose(pa2, pa + dth)


def test_rotate_radec_covariance():
    # rotating a circular error ellipse leaves it unchanged
    out = prec.rotate_radec(10.0, 20.0, 5.0, 1.0, 1.0, 0.0)
    assert np.allclose(out[2:], [1.0, 1.0, 0.0])
    # covariance determinant is invariant under rotation
    sx, sy, r = 2.0, 0.5, 0.3
    _, _, sx2, sy2, r2 = prec.rotate_radec(1.0, 1.0, 20.0, sx, sy, r)
    det1 = sx**2 * sy**2 * (1 - r**2)
    det2 = sx2**2 * sy2**2 * (1 - r2**2)
    assert np.isclose(det1, det2)


def _make_table():
    from orbitize import read_input

    csv = (
        "epoch,object,quant1,quant1_err,quant2,quant2_err,quant12_corr,quant_type,instrument\n"
        "15020.0,1,1500.0,30.0,200.0,0.5,nan,seppa,visual\n"      # ~1900
        "51544.5,1,1400.0,5.0,210.0,0.1,nan,seppa,speckle\n"      # 2000.0
        "58849.0,1,400.0,1.0,-1300.0,1.0,0.2,radec,gpi\n"      # 2020
        "47892.0,1,800.0,10.0,900.0,10.0,nan,radec,speckle\n"     # 1990
        "58849.0,0,1.2,0.1,nan,nan,nan,rv,hires\n"
    )
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w") as f:
        f.write(csv)
    tab = read_input.read_file(path)
    os.remove(path)
    return tab


def test_correct_data_table():
    tab = _make_table()
    ra, dec = 86.82, -51.07
    out = prec.correct_data_table(
        tab, ra, dec, equinox="date", instruments=["visual", "speckle"], verbose=False
    )
    # input untouched
    assert tab["quant2"][0] == 200.0
    t_yr = prec.mjd_to_julian_year(15020.0)
    expected = 200.0 + prec.pa_precession_correction_rigorous(ra, dec, t_yr, 2000.0)[0]
    assert np.isclose(out["quant2"][0], expected)
    heintz = 200.0 - 0.00557 * np.sin(np.radians(ra)) / np.cos(np.radians(dec)) * (t_yr - 2000)
    assert abs(out["quant2"][0] - heintz) < 0.01
    # J2000.0 epoch speckle row: zero correction
    assert np.isclose(out["quant2"][1], 210.0)
    # GPI row not selected -> unchanged
    assert out["quant1"][2] == 400.0 and out["quant2"][2] == -1300.0
    assert out["pa_precession_corr"][2] == 0.0
    # 1990 speckle radec row rotated, separation preserved
    s_in = np.hypot(800.0, 900.0)
    s_out = np.hypot(out["quant1"][3], out["quant2"][3])
    assert np.isclose(s_in, s_out)
    assert out["pa_precession_corr"][3] != 0.0
    # rv untouched
    assert out["quant1"][4] == 1.2
    # double application guarded
    with pytest.raises(RuntimeError):
        prec.correct_data_table(out, ra, dec, verbose=False)


def test_rows_and_column_equinox():
    tab = _make_table()
    tab["eq"] = ["date", "B1950", "J2000", "date", "J2000"]
    out = prec.correct_data_table(tab, 90.0, 30.0, equinox="eq", rows=[0, 1], verbose=False)
    assert out["pa_precession_corr"][0] != 0
    # row 1 measured at 2000.0 but referred to B1950: correction is non-zero
    assert out["pa_precession_corr"][1] != 0
    assert np.all(out["pa_precession_corr"][2:] == 0)


def test_rigorous_table_path():
    pytest.importorskip("astropy")
    tab = _make_table()
    a = prec.correct_data_table(tab, 86.82, -51.07, method="heintz", verbose=False)
    b = prec.correct_data_table(tab, 86.82, -51.07, verbose=False)
    assert b.meta["precession_method"] == "rigorous"
    assert np.allclose(a["pa_precession_corr"], b["pa_precession_corr"], atol=3e-3)


def test_default_is_rigorous_and_heintz_warns():
    tab = _make_table()
    out = prec.correct_data_table(tab, 90.0, 30.0, verbose=False)
    assert out.meta["precession_method"] == "rigorous"
    with pytest.warns(UserWarning, match="first-order"):
        prec.pa_precession_correction(90.0, 30.0, 2010.0)
    with pytest.warns(UserWarning, match="first-order"):
        prec.correct_data_table(tab, 90.0, 30.0, method="heintz", verbose=False)
    assert not hasattr(prec, "pa_proper_motion_correction")


# Published per-star PA precession rates (deg per 100 yr, rounded to 0.1) from
# Bailey et al. 2024, MNRAS 535, 1586, Table A1 (van de Kamp 1967 formula applied
# by the authors). J2000 coordinates in (deg, deg).
_BAILEY2024 = [
    # HD,     RA,     Dec,   rate
    (7927, 20.02, 58.23, +0.4), (23512, 56.64, 23.62, +0.5), (43384, 94.24, 23.74, +0.6),
    (80558, 139.68, -51.56, +0.6), (84810, 146.31, -62.51, +0.7), (111613, 192.82, -60.33, -0.2),
    (147084, 245.16, -24.17, -0.6), (149757, 249.29, -10.57, -0.5), (154445, 256.38, -0.89, -0.5),
    (160529, 265.50, -33.50, -0.7), (161056, 265.95, -7.08, -0.6), (161471, 266.90, -40.13, -0.7),
    (183143, 291.86, 18.30, -0.5), (187929, 298.12, 1.01, -0.5), (198478, 312.23, 46.11, -0.6),
    (203532, 323.48, -82.68, -2.7), (210121, 332.05, -3.53, -0.3),
]


@pytest.mark.parametrize("hd,ra,dec,rate", _BAILEY2024)
def test_rigorous_matches_published_rates(hd, ra, dec, rate):
    pytest.importorskip("astropy")
    got = prec.pa_precession_correction_rigorous(ra, dec, 1920.0, 2020.0)[0]
    assert abs(got - rate) <= 0.06  # published values are rounded to 0.1 deg


def test_published_sco_rate():
    """Ruiz et al. 2019 (arXiv:1903.05330): pairs near 16h30m, -20 deg precess
    by ~0.55 deg per century."""
    pytest.importorskip("astropy")
    got = prec.pa_precession_correction_rigorous(247.5, -20.0, 1900.0, 2000.0)[0]
    assert np.isclose(abs(got), 0.55, atol=0.02)
