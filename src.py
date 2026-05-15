import numpy as np 
import datetime as dt 
import pandas as pd 
from tqdm import tqdm 
from dataclasses import dataclass 

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch 

import astropy 
import astropy.units as u 
import astroquery.simbad
import astroquery.jplhorizons
import astroplan 
import pytz 
import timezonefinder as tzf 










# Create an Observor (either a location, like "Winer", or a latitude/longitude tuple)
def get_observer(observatory_name=None, lat_long_tuple=None):
    """
    Create an astroplan Observer from either a known observatory name
    or a latitude/longitude pair.

    Parameters
    ----------
    observatory_name : str, optional
        Name of a known observatory (e.g., "Winer"). 
        Uses Astropy's site database to look up coordinates. 

    lat_long_tuple : tuple of float, optional
        Latitude and longitude in degrees as (lat, lon).
        Example: (51.2, -91.7)

    Returns
    -------
    astroplan.Observer
        Observer object with location and timezone set.

    Examples
    --------
    >>> get_observer(observatory_name="Winer")
    >>> get_observer(lat_long_tuple=(51.2, -91.7))
    """

    if (observatory_name is None and lat_long_tuple is None) or (observatory_name is not None and lat_long_tuple is not None): 
        raise ValueError("Must provide either 'observatory_name' or 'lat_long_tuple' (but not both)")
    
    # Look up latitude/longitude coords of a known observatory  
    if observatory_name is not None: 
        obs_loc = astropy.coordinates.EarthLocation.of_site(observatory_name)
        lat_long_tuple = (obs_loc.lat.deg, obs_loc.lon.deg)

    location = astropy.coordinates.EarthLocation(
        lat=lat_long_tuple[0]*u.deg,
        lon=lat_long_tuple[1]*u.deg,
        height=0*u.m) 
    
    # Use latitude/longitude provided and look up the local timezone at that location 
    timezone_str = tzf.TimezoneFinder().timezone_at(lat=lat_long_tuple[0], lng=lat_long_tuple[1])

    # Include observatory name in Observer object (i.e., "Winer")
    Observer = astroplan.Observer(location=location, timezone=timezone_str)
    Observer.name = observatory_name 

    # Save info string that can be included in the plot title 
    lat = Observer.location.lat.deg
    lon = Observer.location.lon.deg
    lat_str = f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon):.2f}° {'E' if lon >= 0 else 'W'}"
    observer_coord_str = f"{lat_str}, {lon_str}" 
    Observer.coord_str = observer_coord_str
    
    return Observer 










@dataclass 
class Target: 
    ra: np.array 
    dec: np.array 
    times: np.array 
    name: str = "" 
    coord_str: str = "RA/dec unavailable" 





def get_target(target_name=None, target_radec_str=None):
    """
    Resolve a target's RA/dec coordinates for use in visibility calculations.

    Accepts either a target name (looked up automatically) or a direct RA/dec
    string. If a name is provided, Simbad is tried first (for fixed targets like
    stars and galaxies), falling back to JPL Horizons for solar system bodies.
    In both cases, returns a daily RA/dec array spanning 2 years from 2025-12-31.

    Parameters
    ----------
    target_name : str, optional
        Name of the target, e.g. "Vega", "M31", "Ceres", or a JPL Horizons
        numeric ID such as 599 for Jupiter. Mutually exclusive with
        target_radec_str.
    target_radec_str : str, optional
        RA/dec string in the format "HH MM SS +DD MM SS", e.g.
        "05 34 32.0 +22 00 52". Mutually exclusive with target_name.

    Returns
    -------
    Target
        A Target object with the following attributes:
            - ra : np.ndarray of RA values in degrees, one per day
            - dec : np.ndarray of Dec values in degrees, one per day
            - times : pd.DatetimeIndex of UTC timestamps, one per day
            - name : str, target name (if provided)
            - coord_str : str, formatted RA/dec string for plot titles
              (only set for fixed targets; moving targets change position
              over time so no single coordinate string is meaningful)

    Raises
    ------
    ValueError
        If both or neither of target_name and target_radec_str are provided.
        If target_name is ambiguous in JPL Horizons (e.g. "Jupiter" instead
        of the numeric ID 599).
        If the target cannot be resolved in either Simbad or JPL Horizons.

    Examples
    --------
    >>> target = get_target(target_name="Vega")
    >>> target = get_target(target_name=599)          # Jupiter by JPL ID
    >>> target = get_target(target_name="Ceres")      # falls back to Horizons
    >>> target = get_target(target_radec_str="05 34 32.0 +22 00 52")
    """

    # Helper func: format a coord nicely for plot title 
    def format_coord(ra, dec): 
        coord = astropy.coordinates.SkyCoord(ra=ra, dec=dec, unit="deg") 
        coord_str = coord.to_string('hmsdms', precision=1) 
        coord_str = coord_str.replace("h", ":")
        coord_str = coord_str.replace("m", ":", 1)
        coord_str = coord_str.replace("s", "", 1)
        coord_str = coord_str.replace("d", "°")
        coord_str = coord_str.replace("m", "\'")
        coord_str = coord_str.replace("s", "\"")
        return coord_str 
    
    # Calculate ra/dec of target once per day from December 31st until num_yrs (1 or 2) years ahead 
    num_yrs = 2 
    radec_times = pd.date_range(
        start='2025-12-31 12:00:00',
        end=f'{2025+num_yrs}-12-31 12:00:00',
        freq='1D', # 3D = Calculate visibility every 3rd day, 10D = every 10th day, etc 
        tz="UTC", 
    )
    
    # Validate inputs
    if (target_name is None) == (target_radec_str is None):
        raise ValueError("Must provide either 'target_name' or 'target_radec_str' (but not both)")



    # RA/dec provided directly - always fixed
    if target_radec_str is not None:
        coord = astropy.coordinates.SkyCoord(target_radec_str, unit=(u.hourangle, u.deg))
        print(f"Used target_radec_str = {target_radec_str} to generate target coordinates")

        # Create array of repeated values so that it matches the moving output 
        ra_arr = np.full(len(radec_times), coord.ra.deg)
        dec_arr = np.full(len(radec_times), coord.dec.deg) 

        # Return Target object with formatted coord string but no name 
        target_coord_str = format_coord(ra_arr[0], dec_arr[0])
        return Target(ra=ra_arr, dec=dec_arr, times=radec_times, coord_str=target_coord_str)



    # Target name provided - try Simbad first, then Horizons
    if target_name is not None:

        # --- Try Simbad (fixed stars, galaxies, etc.) ---
        try:
            print(f"Looking up '{target_name}' in Simbad...")
            result = astroquery.simbad.Simbad.query_object(target_name)
            if result is None:
                raise ValueError("Simbad returned no results")
            ra = float(result["ra"][0])
            dec = float(result["dec"][0])

            # Create array of repeated values so that it matches the moving output 
            ra_arr = np.full(len(radec_times), ra) 
            dec_arr = np.full(len(radec_times), dec)

            print(f"Retrieved '{target_name}' from Simbad") 
            
            # Return Target object with name and formatted coord str 
            target_coord_str = format_coord(ra_arr[0], dec_arr[0])
            return Target(ra=ra_arr, dec=dec_arr, times=radec_times, coord_str=target_coord_str, name=target_name)



        except Exception as e:
            print(f"Simbad lookup failed ({e}), trying JPL Horizons...")

        # --- Try Horizons (solar system bodies) ---
        try:
            jds = astropy.time.Time(radec_times).jd.tolist()
            ra, dec = [], []
            batch_size = 50 
            for i in range(0, len(jds), batch_size):
                batch = jds[i:i + batch_size]
                eph = astroquery.jplhorizons.Horizons(id=target_name, epochs=batch).ephemerides()
                ra.extend(eph["RA"])
                dec.extend(eph["DEC"])
            print(f"Retrieved '{target_name}' from JPL Horizons") 
            ra_arr = np.array(ra) 
            dec_arr = np.array(dec) 

            # Return Target object with name but no formatted coord str (because it moves) 
            return Target(ra=ra_arr, dec=dec_arr, times=radec_times, name=target_name)



        except Exception as e:
            # Catch the ambiguous name error specifically and give a helpful message
            if "Ambiguous" in str(e):
                raise ValueError(
                    f"Ambiguous target name '{target_name}' in JPL Horizons. "
                    f"Try using a numeric ID instead (e.g. 599 for Jupiter, 499 for Mars). "
                    f"Full error: {e}"
                ) from None
            raise ValueError(f"Could not resolve '{target_name}' in Simbad or JPL Horizons: {e}") from None










def calc_visibility(observer, target):
    """
    Calculate the altitude of a target and the Sun over time for a given observer.

    Interpolates the target's RA/dec from a daily grid onto a finer 10-minute
    grid, then computes altitude in a single vectorized operation. Works for
    both fixed targets (where RA/dec is constant) and moving solar system bodies
    (where RA/dec varies over time).

    Parameters
    ----------
    observer : Observer
        Observer object containing the observer's location.
    target : Target
        Target object containing:
            - ra : np.ndarray of RA values in degrees, one per day
            - dec : np.ndarray of Dec values in degrees, one per day
            - times : pd.DatetimeIndex of UTC timestamps, one per day

    Returns
    -------
    datetimes_utc_1d : pd.DatetimeIndex
        UTC timestamps at 10-minute intervals spanning the full date range.
    target_alt_1d : np.ndarray
        Altitude of the target in degrees at each timestamp.
    sun_alt_1d : np.ndarray
        Altitude of the Sun in degrees at each timestamp.

    Notes
    -----
    Interpolating RA/dec linearly over 1-day intervals introduces negligible
    error for most solar system bodies. The Moon is an exception and may
    require a finer RA/dec grid for accurate results.
    """
    # Interpolate RA/dec onto altitude times grid (10 minute spacing) 
    print("Interpolating RA/dec onto 10 minute grid")
    datetimes_utc_1d = pd.date_range(target.times[0], target.times[-1], freq="10min")
    jds_daily = astropy.time.Time(target.times).jd
    jds_alt = astropy.time.Time(datetimes_utc_1d).jd
    ra_interp = np.interp(jds_alt, jds_daily, target.ra)
    dec_interp = np.interp(jds_alt, jds_daily, target.dec)

    # Single vectorized altitude calculation
    print("Calculating altaz_frame")
    altaz_frame = astropy.coordinates.AltAz(obstime=astropy.time.Time(datetimes_utc_1d), location=observer.location)

    print("Calculating target altitudes")
    target_coords = astropy.coordinates.SkyCoord(ra=ra_interp, dec=dec_interp, unit="deg")
    target_alt_1d = target_coords.transform_to(altaz_frame).alt.to_value()

    print("Calculating Sun altitudes")
    sun_alt_1d = astropy.coordinates.get_sun(astropy.time.Time(datetimes_utc_1d)).transform_to(altaz_frame).alt.to_value()

    return datetimes_utc_1d, target_alt_1d, sun_alt_1d 










def reshape_altitude(observer, times_1d, alt_1d):
    """
    Reshape a 1D array of altitudes into a 2D grid suitable for pcolormesh.

    Converts UTC times to the observer's local timezone, then maps each
    (date, time-of-day) pair to a position in a 2D grid where the x axis
    is date and the y axis is local time running from noon to noon rather
    than midnight to midnight. This noon-to-noon convention keeps each
    observing night in a single contiguous column rather than split across
    two calendar dates.

    Daylight savings time transitions are handled gracefully: 
    during spring-forward, the one missing hour maps to NaN in the grid; 
    during fall-back, the duplicate hour is overwritten with the later value. 

    Parameters
    ----------
    observer : Observer
        Observer object containing the observer's timezone.
    times_1d : pd.DatetimeIndex
        UTC timestamps at regular intervals (e.g. 10-minute spacing).
    alt_1d : np.ndarray
        Altitude values in degrees, one per timestamp in times_1d.

    Returns
    -------
    x_dates_1d : pd.DatetimeIndex
        Unique local dates, timezone-aware, for use as the pcolormesh x axis.
    y_times_1d : np.ndarray of pd.Timestamp
        Unique local times-of-day expressed as timestamps on an arbitrary
        reference date, running from noon to noon, for use as the pcolormesh
        y axis.
    alt_2d : np.ndarray
        2D array of shape (n_times_per_day, n_dates) containing altitude
        values in degrees. NaN where no data exists (e.g. DST spring-forward).

    Notes
    -----
    The reference date used for y_times_1d is arbitrary since only the time
    component is displayed on the plot axis.
    """
    # Convert to local time 
    local_times = times_1d.tz_convert(observer.timezone)

    # Create 1d arrays of dates and times (x and y axes on plot)
    unique_dates = np.unique(local_times.date)
    unique_times = np.unique(local_times.time) 

    # Roll times array over so that it goes noon-noon instead of midnight-midnight
    noon_idx = np.where(unique_times == dt.time(12, 0))[0][0]
    unique_times = np.roll(unique_times, -noon_idx)


    # Build lookup dictionaries for fast indexing
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    hour_to_idx = {t: i for i, t in enumerate(unique_times)}

    # Initialize grid with NaN
    alt_2d = np.full((len(unique_times), len(unique_dates)), np.nan)

    # Fill grid
    for i, (lt, alt) in enumerate(zip(local_times, alt_1d)):
        row = hour_to_idx.get(lt.time())
        col = date_to_idx.get(lt.date())
        if row is not None and col is not None:
            alt_2d[row, col] = alt

    # Convert unique_dates and unique_times to things that can be plotted 
    x_dates_1d = pd.DatetimeIndex(unique_dates, tz=str(observer.timezone))
    # Reference date is arbitrary since we're only showing the time on the plot anyway 
    ref_date = pd.Timestamp("2025-12-15 00:00:00", tz=str(observer.timezone))  
    y_times_1d = np.array([
        ref_date + dt.timedelta(hours=t.hour + (24 if t.hour < 12 else 0), minutes=t.minute, seconds=t.second)
        for t in unique_times
    ])
    return x_dates_1d, y_times_1d, alt_2d 










def plot_visibility(
        x_dates_1d, 
        y_times_1d, 
        target_alt_2d, 
        sun_alt_2d, 
        target, 
        observer, 
        target_min_alt=25, 
        sun_max_alt=-6
    ):
    """
    Plot a 2D visibility map showing when a target is observable over a two-year period.

    The x axis represents date and the y axis represents local time of day,
    running from noon to noon to keep each observing night in a single contiguous
    column. The plot consists of three layered components:

    1. **Twilight background**
       Colored bands indicating solar altitude ranges:
       - Full night:              Sun < -18°
       - Astronomical twilight:  -18° ≤ Sun < -12°
       - Nautical twilight:      -12° ≤ Sun < -6°
       - Civil twilight:          -6° ≤ Sun < 0°
       - Civil daylight:           0° ≤ Sun < 6°
       - Nautical daylight:        6° ≤ Sun < 12°
       - Astronomical daylight:   12° ≤ Sun < 18°
       - Full day:               Sun ≥ 18°

    2. **Target visibility mask**
       Semi-transparent black overlay showing when the target is above
       `target_min_alt`, regardless of Sun position. A solid black contour
       marks the boundary.

    3. **Target altitude colormap**
       Grayscale shading showing the target's altitude, masked to only appear
       where both conditions are met:
       - target_alt > target_min_alt (target is above minimum altitude)
       - sun_alt <= sun_max_alt (Sun is below the maximum allowed altitude)
       A red contour marks the boundary of this region, and a dashed green
       line on the colorbar indicates the target's maximum altitude.

    Parameters
    ----------
    x_dates_1d : pd.DatetimeIndex
        Timezone-aware dates for the x axis, one per day.
    y_times_1d : np.ndarray of pd.Timestamp
        Local times-of-day for the y axis, running noon to noon, expressed
        as timestamps on an arbitrary reference date.
    target_alt_2d : np.ndarray
        2D array of shape (n_times_per_day, n_dates) containing target
        altitude in degrees.
    sun_alt_2d : np.ndarray
        2D array of shape (n_times_per_day, n_dates) containing Sun
        altitude in degrees.
    target : Target
        Target object providing target.name and target.coord_str for the
        plot title.
    observer : Observer
        Observer object providing observer.name, observer.coord_str, and
        observer.timezone for the plot title and y axis label.
    target_min_alt : float, optional
        Minimum target altitude in degrees to consider the target observable.
        Default is 25°.
    sun_max_alt : float, optional
        Maximum Sun altitude in degrees to consider it sufficiently dark.
        Default is -6° (civil twilight).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The completed figure, which can be saved or displayed by the caller.

    Example
    -------
    >>> datetimes_utc_1d, target_alt_1d, sun_alt_1d = src.calc_visibility(observer, target)
    >>> x_dates, y_times, sun_alt_2d = src.reshape_altitude(observer, datetimes_utc_1d, sun_alt_1d)
    >>> x_dates, y_times, target_alt_2d = src.reshape_altitude(observer, datetimes_utc_1d, target_alt_1d)
    >>> fig = src.plot_visibility(x_dates, y_times, target_alt_2d, sun_alt_2d, target, observer)
    """

    # Create figure 
    fig, ax = plt.subplots(figsize=(16,10))

    # X-axis: Dates   
    def format_date(x, pos=None):
        date = mdates.num2date(x)
        if date.month in [1, 12]:
            return date.strftime('%b \n%Y')  # Jan 2026, Dec 2026, Jan 2027, Dec 2027 
        else:
            return date.strftime('%b')     # Feb, Mar, Apr, ...
    ax.set_xlabel("Date of observation") 
    ax.set_xlim(x_dates_1d[0], x_dates_1d[-1]) 
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    ax.xaxis.set_major_formatter(format_date)

    # Y-axis: Local time  
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(str(observer.timezone)))) 
    ax.set_ylabel(f"Local time ({str(observer.timezone)})") 
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=2)) 
    ax.yaxis.set_minor_locator(mdates.HourLocator(interval=1)) 

    # Title    
    ax.set_title(f"Observer: {observer.name} ({observer.coord_str}) \nTarget: {target.name} ({target.coord_str})")



    # 1: Twilight levels (ignore target)
    levels = [-90, -18, -12, -6, 0, 6, 12, 18, 90]
    base_night = mcolors.to_rgba("cornflowerblue")
    base_day = mcolors.to_rgba("gold")
    colors = [
        (*base_night[:3], 0.9),
        (*base_night[:3], 0.7),
        (*base_night[:3], 0.5),
        (*base_night[:3], 0.3),
        (*base_day[:3], 0.3),
        (*base_day[:3], 0.5),
        (*base_day[:3], 0.7),
        (*base_day[:3], 0.9),
    ]

    # Interior: blue to yellow background gradient 
    ax.contourf(
        x_dates_1d,
        y_times_1d,
        sun_alt_2d, 
        levels=levels,
        colors=colors, 
    )

    # Borders: higher zorder, so that it plots the borders on top of the target altitude colormap 
    # (So you can tell which level of twilight you're in while also seeing the altitude)
    # Replace the border between civil twilight and civil daylight with white 
    # (Makes it easier to see the distinction between night and day)
    colors_borders = [
        (*base_night[:3], 0.9),
        (*base_night[:3], 0.7),
        (*base_night[:3], 0.5),
        (*base_night[:3], 0.3),
        "white",
        (*base_day[:3], 0.5),
        (*base_day[:3], 0.7),
        (*base_day[:3], 0.9),
    ]
    ax.contour(
        x_dates_1d, 
        y_times_1d, 
        sun_alt_2d, 
        levels=levels, 
        colors=colors_borders, 
        linewidths=1, 
        linestyles="solid", 
        zorder=5, 
    )



    # 2: Target visibility binary (ignore sun, just show if the target is up or not)

    # Interior: use low alpha so that you can see the twilight colors through it 
    levels = [target_min_alt, 90] 
    colors = [(*mcolors.to_rgba("black")[:3], 0.2)]
    ax.contourf(
        x_dates_1d,
        y_times_1d,
        target_alt_2d,  
        levels=levels,
        colors=colors, 
    )

    # Border: 
    ax.contour(
        x_dates_1d,
        y_times_1d,
        target_alt_2d,  
        levels=levels,
        colors="black", 
        linewidths=1, 
        zorder=5, 
    )



    # 3: Target altitude colormap 

    # Display altitude only where target is up and sun is down 
    visible = (target_alt_2d > target_min_alt) & (sun_alt_2d <= sun_max_alt)
    Z = np.ma.MaskedArray(target_alt_2d, mask=~visible)
    dark_greys = mcolors.LinearSegmentedColormap.from_list("dark_greys",cm.Greys(np.linspace(0.3, 1, 256)))
    target_alt_plot = ax.pcolormesh(
        x_dates_1d,
        y_times_1d,
        Z, 
        cmap=dark_greys, 
        zorder=4, 
        vmin=0, 
        vmax=90, 
    )

    # Add red contour around visibile region to make it pop more 
    ax.contour(
        x_dates_1d,
        y_times_1d,
        visible.astype(int),
        levels=[0.5],
        colors="red",
        linewidths=2,
        zorder=6
    )

    # Color bar 
    cbar = plt.colorbar(target_alt_plot, pad=0.01, label="Target altitude (deg)") 
    cbar.ax.set_ylim(target_min_alt, 90) 
    cbar.ax.axhline(np.nanmax(Z), color="limegreen", ls="dashed") # Show where max altitude is on color bar 

    # Create "fake"/unplotted objects to display in the legend 
    legend_elements = [
        Patch(
            facecolor=(0, 0, 0, 0.2),  # RGBA: transparent fill only
            edgecolor="black",
            linewidth=1.5,
            label=f"Target >{target_min_alt}°"
        ), 
        Line2D([0], [0], color='red', lw=2, label=f"Target >{target_min_alt}° \nand Sun <{sun_max_alt}°"),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return fig 













