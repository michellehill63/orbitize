"""
orbitize.precession - contributed by Michelle L Hill
===================

Correct relative astrometry (position angles, and the RA/Dec offsets
derived from them) for the precession of the Earth's spin axis.

Background
----------
A position angle (PA) is measured from the direction of the north celestial
pole, through east. Because the pole precesses (period ~25,800 yr), "north"
at a given star points in a slightly different direction at each epoch.
A PA referred to the *equinox of date* (the normal case for historical
filar-micrometer, and many speckle, measures) therefore differs from a PA
referred to a standard equinox such as J2000.0.

The default method ("rigorous") rotates the local north direction with
astropy's IAU 1976 FK5 equinox transformation. The classical first-order reduction 
``method="heintz"`` is for cross-checks and is based on 
Heintz (1978, *Double Stars*, D. Reidel, Ch. 4, eq. 5):

    theta_0 = theta - 0.00557 deg * sin(alpha) * sec(delta) * (t - t0)

where ``t`` is the equinox the measure refers to and ``t0`` is the target
equinox, both in years. The coefficient is the annual general precession in
declination, n ~= 20.04 arcsec/yr = 0.00557 deg/yr.

Equivalently, the PA rotates at d(theta)/dt = +n sin(alpha) sec(delta), so
the pole-of-date PA leads the J2000 PA for 0 < alpha < 180 deg and lags it
for 180 < alpha < 360 deg. 

Size of the effect, per decade:
    * alpha =  90 deg, delta = +30 deg  ->  +0.064 deg
    * alpha =  90 deg, delta = +60 deg  ->  +0.111 deg
    * alpha =  90 deg, delta = +80 deg  ->  +0.321 deg
    * alpha = 0 or 180 deg               ->   0 (to first order)


Note: Only measures whose PAs refer to the "pole of date" need this correction. Modern imaging
astrometry calibrated against ICRS/J2000 astrometric fields (e.g. GPI,
SPHERE, NIRC2, and VLTI/GRAVITY with standard reductions) and Gaia are
already on a fixed J2000/ICRS frame and must NOT be corrected. This is
why the table-level function lets you choose rows by instrument or row index,
or give an equinox per row.


"""

import warnings

import numpy as np

__all__ = [
    "PRECESSION_RATE_DEG_PER_YR",
    "mjd_to_julian_year",
    "parse_equinox",
    "pa_precession_correction",
    "pa_precession_correction_rigorous",
    "precess_pa",
    "rotate_radec",
    "correct_data_table",
    "read_file",
]

#: Annual general precession in declination, n, in degrees per year.
#: n = 20.04 arcsec/yr (IAU 1976, J2000), so n = 0.005567 deg/yr. Heintz
#: (1978) eq. 5 uses 0.00557.
PRECESSION_RATE_DEG_PER_YR = 0.00557

_MJD_J2000 = 51544.5  # MJD of J2000.0 (JD 2451545.0)
_DAYS_PER_JULIAN_YEAR = 365.25

# Name of the column added to the data table to record what was applied.
CORR_COLUMN = "pa_precession_corr"
_META_FLAG = "precession_corrected"


def mjd_to_julian_year(mjd):
    """Convert MJD to a Julian epoch (e.g. 51544.5 -> 2000.0).

    Args:
        mjd (float or np.array): Modified Julian Date(s).

    Returns:
        float or np.array: Julian year(s).
    """
    return 2000.0 + (np.asarray(mjd, dtype=float) - _MJD_J2000) / _DAYS_PER_JULIAN_YEAR


def parse_equinox(equinox):
    """Turn an equinox spec into a decimal year.

    Accepts floats (``2000.0``), Julian/Besselian strings (``"J2000"``,
    ``"B1950"``), or an ``astropy.time.Time``. Besselian epochs are
    converted with astropy if available; otherwise the value after "B" is
    used directly (the difference is < 0.001 yr at B1950, so it does not
    matter for this correction).

    Args:
        equinox (float, str or astropy.time.Time): equinox to parse.

    Returns:
        float: equinox as a decimal year.
    """
    if hasattr(equinox, "jyear"):  # astropy Time
        return float(equinox.jyear)
    if isinstance(equinox, str):
        s = equinox.strip().upper()
        if s.startswith("J"):
            return float(s[1:])
        if s.startswith("B"):
            try:
                from astropy.time import Time

                return float(Time(s, format="byear_str").jyear)
            except Exception:
                return float(s[1:])
        return float(s)
    return float(equinox)



def pa_precession_correction(ra, dec, epoch_from, epoch_to=2000.0, warn=True):
    """Correction to ADD to a PA to move it from one equinox to another.

    Implements Heintz (1978) eq. 5, linear formula is useful to check against the default astropy method
    (see pa_precession_correction_rigorous).:

        delta_theta = theta(epoch_to) - theta(epoch_from)
                    = -0.00557 * sin(alpha) * sec(delta) * (epoch_from - epoch_to)

    (in degrees, epochs in years). For the usual case, a measure in the
    equinox of date ``t`` reduced to J2000, this gives
    ``theta_2000 = theta_t - 0.00557 sin(alpha) sec(delta) (t - 2000)``.

    Args:
        ra (float or np.array): right ascension of the system [deg].
            J2000/ICRS values are fine; using the coordinates of date instead
            changes the result only at second order.
        dec (float or np.array): declination of the system [deg].
        epoch_from (float or np.array): equinox the PA currently refers to
            [decimal year].
        epoch_to (float or np.array): equinox to refer the PA to
            [decimal year]. Default 2000.0.
        warn (bool): emit the "first-order approximation" warning.

    Returns:
        float or np.array: correction [deg] to add to the PA.
    """
    if warn:
        warnings.warn(
            "pa_precession_correction uses the first-order Heintz (1978) eq. 5 "
            "approximation; pa_precession_correction_rigorous (astropy) is the "
            "default and is exact. Pass warn=False to silence.",
            stacklevel=2,
        )
    ra_r = np.radians(np.asarray(ra, dtype=float))
    dec_r = np.radians(np.asarray(dec, dtype=float))
    cosd = np.cos(dec_r)
    dt = np.asarray(epoch_from, dtype=float) - np.asarray(epoch_to, dtype=float)
    dtheta = -PRECESSION_RATE_DEG_PER_YR * np.sin(ra_r) / cosd * dt
    # The linear formula ignores the change of (alpha, delta) over dt. That
    # error grows as tan(delta) * (n dt): for Polaris over a century it is
    # 26.5 deg (linear) vs 15.9 deg (rigorous).
    if np.any(np.abs(np.degrees(dec_r)) > 80.0) or np.any(np.abs(dtheta) > 1.0):
        warnings.warn(
            "|dec| > 80 deg or |dPA| > 1 deg: the first-order Heintz (1978) "
            "eq. 5 precession correction may be inaccurate here. Use "
            "method='rigorous' (requires astropy)."
        )
    return dtheta


def pa_precession_correction_rigorous(ra, dec, epoch_from, epoch_to=2000.0):
    """

    Default method. Uses the IAU 1976 precession model through astropy's FK5 equinox
    transformation. It measures how far the direction of the pole of
    ``epoch_from`` is rotated, at the star's position, relative to the pole
    of ``epoch_to``. Useful for very long baselines, targets near the poles.

    Args:
        ra, dec (float or np.array): J2000/ICRS coordinates of the system
            [deg].
        epoch_from (float or np.array): equinox the PA currently refers to
            [Julian year].
        epoch_to (float): equinox to refer the PA to [Julian year].

    Returns:
        np.array: correction [deg] to add to the PA.
    """
    import astropy.units as u
    from astropy.coordinates import FK5, SkyCoord
    from astropy.time import Time

    ra = np.atleast_1d(np.asarray(ra, dtype=float))
    dec = np.atleast_1d(np.asarray(dec, dtype=float))
    epoch_from = np.atleast_1d(np.asarray(epoch_from, dtype=float))
    ra, dec, epoch_from = np.broadcast_arrays(ra, dec, epoch_from)

    star_j2000 = SkyCoord(ra * u.deg, dec * u.deg, frame=FK5(equinox="J2000"))
    frame_to = FK5(equinox=Time(float(parse_equinox(epoch_to)), format="jyear"))
    star_to = star_j2000.transform_to(frame_to)

    out = np.empty(ra.shape, dtype=float)
    # FK5 frames take a scalar equinox, so loop over the unique epochs.
    for ep in np.unique(epoch_from):
        sel = epoch_from == ep
        frame_from = FK5(equinox=Time(float(ep), format="jyear"))
        star_from = star_j2000[sel].transform_to(frame_from)
        # A point 1 arcsec due north of the star in the "from" frame
        north_from = star_from.directional_offset_by(0.0 * u.deg, 1.0 * u.arcsec)
        # Its PA in the "to" frame is the rotation between the two norths.
        pa_to = star_to[sel].position_angle(north_from.transform_to(frame_to)).deg
        out[sel] = (pa_to + 180.0) % 360.0 - 180.0
    return out


def precess_pa(pa, ra, dec, epoch_from, epoch_to=2000.0, method="rigorous"):
    """Move position angle(s) from one equinox to another.

    Args:
        pa (float or np.array): position angle(s) [deg], east of north.
        ra, dec (float): system coordinates [deg].
        epoch_from (float or np.array): equinox the PAs refer to [yr].
        epoch_to (float): target equinox [yr]. Default 2000.0.
        method (str): ``"rigorous"`` (default; IAU 1976 precession via
            astropy) or ``"heintz"`` (first-order Heintz 1978 eq. 5).

    Returns:
        np.array: precessed PA(s) in [0, 360) deg.
    """
    if method == "heintz":
        dtheta = pa_precession_correction(ra, dec, epoch_from, epoch_to)
    elif method == "rigorous":
        dtheta = pa_precession_correction_rigorous(ra, dec, epoch_from, epoch_to)
    else:
        raise ValueError("method must be 'heintz' or 'rigorous', got {0!r}".format(method))
    return np.mod(np.asarray(pa, dtype=float) + dtheta, 360.0)


def rotate_radec(raoff, decoff, dtheta, raoff_err=None, decoff_err=None, corr=None):
    """Rotate (RA, Dec) offsets by the same angle as a PA correction.

    With orbitize's convention (RA offset = sep sin PA, Dec offset =
    sep cos PA), adding ``dtheta`` to the PA is equivalent to

        raoff'  =  raoff cos(dtheta) + decoff sin(dtheta)
        decoff' = -raoff sin(dtheta) + decoff cos(dtheta)

    Uncertainties are propagated by rotating the full 2x2 covariance
    matrix, so the correlation coefficient is updated as well.

    Args:
        raoff, decoff (float or np.array): offsets [mas].
        dtheta (float or np.array): PA correction [deg].
        raoff_err, decoff_err (float or np.array, optional): 1-sigma errors.
        corr (float or np.array, optional): RA/Dec correlation coefficient;
            NaN/None is treated as 0.

    Returns:
        tuple: (raoff', decoff') or, if errors are given,
        (raoff', decoff', raoff_err', decoff_err', corr').
    """
    t = np.radians(np.asarray(dtheta, dtype=float))
    c, s = np.cos(t), np.sin(t)
    x = np.asarray(raoff, dtype=float)
    y = np.asarray(decoff, dtype=float)
    x_new = c * x + s * y
    y_new = -s * x + c * y
    if raoff_err is None or decoff_err is None:
        return x_new, y_new

    sx = np.asarray(raoff_err, dtype=float)
    sy = np.asarray(decoff_err, dtype=float)
    rho = np.zeros_like(sx) if corr is None else np.nan_to_num(np.asarray(corr, dtype=float))
    cxx, cyy, cxy = sx**2, sy**2, rho * sx * sy
    # C' = R C R^T with R = [[c, s], [-s, c]]
    cxx_n = c**2 * cxx + 2 * c * s * cxy + s**2 * cyy
    cyy_n = s**2 * cxx - 2 * c * s * cxy + c**2 * cyy
    cxy_n = -c * s * cxx + (c**2 - s**2) * cxy + c * s * cyy
    sx_n, sy_n = np.sqrt(cxx_n), np.sqrt(cyy_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_n = np.where((sx_n > 0) & (sy_n > 0), cxy_n / (sx_n * sy_n), 0.0)
    return x_new, y_new, sx_n, sy_n, rho_n


# orbitize data-table interface
# ---------------------------------------------------------------------------
def _resolve_equinox(data_table, equinox, epochs_yr):
    """Return a per-row array of the equinox (decimal years) each row is in."""
    n = len(data_table)
    if isinstance(equinox, str) and equinox.lower() == "date":
        return epochs_yr.copy()
    if isinstance(equinox, str) and equinox in data_table.colnames:
        col = data_table[equinox]
        out = np.empty(n)
        for i, val in enumerate(col):
            if isinstance(val, str) and val.strip().lower() == "date":
                out[i] = epochs_yr[i]
            else:
                out[i] = parse_equinox(val)
        return out
    if np.ndim(equinox) == 0:
        return np.full(n, parse_equinox(equinox))
    arr = np.asarray(equinox, dtype=object)
    if arr.shape != (n,):
        raise ValueError("equinox array must have one entry per data-table row")
    return np.array(
        [
            epochs_yr[i] if (isinstance(v, str) and v.lower() == "date") else parse_equinox(v)
            for i, v in enumerate(arr)
        ]
    )


def correct_data_table(
    data_table,
    ra,
    dec,
    equinox="date",
    target_equinox=2000.0,
    rows=None,
    instruments=None,
    method="rigorous",
    inplace=False,
    force=False,
    verbose=True,
):
    """Apply the precession correction to an orbitize data table.

    Works on the table returned by :func:`orbitize.read_input.read_file`
    (columns ``epoch`` [MJD], ``quant1``, ``quant1_err``, ``quant2``,
    ``quant2_err``, ``quant12_corr``, ``quant_type``, ``instrument``).

    * ``seppa`` rows: ``quant2`` (PA) gets the correction added; the
      separation and both errors are unchanged.
    * ``radec`` rows: (``quant1``, ``quant2``) and their covariance are
      rotated by the same angle (see :func:`rotate_radec`).
    * all other rows (``rv``, ``brightness``, ...) are untouched.

    The correction applied to each row (deg; 0 where nothing was applied) is
    stored in a new column ``pa_precession_corr``, and
    ``data_table.meta["precession_corrected"]`` is set so that a second call
    raises an error instead of correcting twice.

    Args:
        data_table (astropy.table.Table): orbitize data table.
        ra, dec (float): system coordinates [deg] (J2000/ICRS).
        equinox (str, float, array or column name): equinox the measured
            PAs refer to.

            * ``"date"`` (default): each row is in the equinox of its own
              observation epoch (typical for visual/speckle measures).
            * a number or ``"J2000"``/``"B1950"``: one equinox for all
              selected rows.
            * an array with one entry per row (numbers or ``"date"``).
            * the name of a column in ``data_table``.
        target_equinox (float or str): equinox to refer all PAs to. Use the
            same equinox as the rest of your data (normally 2000.0, i.e.
            ICRS).
        rows (array of int or bool, optional): only correct these rows.
        instruments (list of str, optional): only correct rows whose
            ``instrument`` value is in this list. Combined with ``rows`` by
            logical AND if both are given.
        method (str): ``"rigorous"`` (default, astropy IAU 1976 precession)
            or ``"heintz"`` (first-order eq. 5; accurate to ~0.01 deg per
            century away from the poles, mainly for cross-checks).
        inplace (bool): modify ``data_table`` in place. Default False (works
            on a copy).
        force (bool): correct again even if the table is already flagged as
            corrected.
        verbose (bool): print a short summary.

    Returns:
        astropy.table.Table: the corrected table.
    """
    if data_table.meta.get(_META_FLAG, False) and not force:
        raise RuntimeError(
            "This data table has already been precession-corrected "
            "(meta['{0}'] is True). Pass force=True to apply it again.".format(_META_FLAG)
        )

    tab = data_table if inplace else data_table.copy()
    n = len(tab)
    epochs_yr = mjd_to_julian_year(np.asarray(tab["epoch"], dtype=float))
    qtype = np.array([str(q) for q in tab["quant_type"]])
    astrometric = (qtype == "seppa") | (qtype == "radec")

    select = astrometric.copy()
    if rows is not None:
        rows = np.asarray(rows)
        m = np.zeros(n, dtype=bool)
        if rows.dtype == bool:
            if rows.shape != (n,):
                raise ValueError("Boolean `rows` mask must have one entry per row")
            m = rows
        else:
            m[rows] = True
        select &= m
    if instruments is not None:
        if "instrument" not in tab.colnames:
            raise ValueError("data_table has no 'instrument' column")
        inst = np.array([str(i) for i in tab["instrument"]])
        select &= np.isin(inst, list(instruments))

    eq_from = _resolve_equinox(tab, equinox, epochs_yr)
    eq_to = parse_equinox(target_equinox)

    dtheta = np.zeros(n)
    if select.any():
        idx = np.where(select)[0]
        if method == "heintz":
            dtheta[idx] = pa_precession_correction(ra, dec, eq_from[idx], eq_to)
        elif method == "rigorous":
            dtheta[idx] = pa_precession_correction_rigorous(ra, dec, eq_from[idx], eq_to)
        else:
            raise ValueError("method must be 'heintz' or 'rigorous'")

    # seppa rows: shift PA
    sp = select & (qtype == "seppa")
    if sp.any():
        pa = np.asarray(tab["quant2"], dtype=float)
        pa[sp] = np.mod(pa[sp] + dtheta[sp], 360.0)
        tab["quant2"][sp] = pa[sp]

    # radec rows: rotate offsets and covariance
    rd = select & (qtype == "radec")
    if rd.any():
        x = np.asarray(tab["quant1"], dtype=float)[rd]
        y = np.asarray(tab["quant2"], dtype=float)[rd]
        sx = np.asarray(tab["quant1_err"], dtype=float)[rd]
        sy = np.asarray(tab["quant2_err"], dtype=float)[rd]
        rho = np.asarray(tab["quant12_corr"], dtype=float)[rd]
        had_corr = np.isfinite(rho)
        x2, y2, sx2, sy2, rho2 = rotate_radec(x, y, dtheta[rd], sx, sy, rho)
        tab["quant1"][rd] = x2
        tab["quant2"][rd] = y2
        tab["quant1_err"][rd] = sx2
        tab["quant2_err"][rd] = sy2
        # Only write a correlation where one existed or the rotation made a
        # non-negligible one (keeps orbitize's NaN = "no correlation" rows).
        rho_out = np.where(had_corr | (np.abs(rho2) > 1e-6), rho2, np.nan)
        tab["quant12_corr"][rd] = rho_out

    if CORR_COLUMN in tab.colnames:
        tab[CORR_COLUMN] = np.asarray(tab[CORR_COLUMN], dtype=float) + dtheta
    else:
        tab[CORR_COLUMN] = dtheta
    tab[CORR_COLUMN].unit = "deg"
    tab.meta[_META_FLAG] = True
    tab.meta["precession_target_equinox"] = eq_to
    tab.meta["precession_method"] = method

    if verbose:
        if select.any():
            print(
                "orbitize.precession: corrected {0} of {1} astrometric rows to equinox "
                "{2:.1f} ({3}); |dPA| range {4:.4f}-{5:.4f} deg.".format(
                    int(select.sum()), int(astrometric.sum()), eq_to, method,
                    np.abs(dtheta[select]).min(), np.abs(dtheta[select]).max(),
                )
            )
        else:
            print("orbitize.precession: no rows selected; table unchanged.")
    return tab


def read_file(filename, ra, dec, **kwargs):
    """Read an orbitize data file and apply the precession correction.

    Convenience wrapper: ``orbitize.read_input.read_file(filename)`` followed
    by :func:`correct_data_table` with ``**kwargs``.

    Args:
        filename (str): path to an orbitize-readable data file.
        ra, dec (float): system coordinates [deg].
        **kwargs: passed to :func:`correct_data_table`.

    Returns:
        astropy.table.Table: corrected orbitize data table.
    """
    from orbitize import read_input

    return correct_data_table(read_input.read_file(filename), ra, dec, **kwargs)
