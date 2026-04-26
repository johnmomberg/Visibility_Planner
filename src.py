import numpy as np 

import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors 
from matplotlib.lines import Line2D
from matplotlib.patches import Patch 

import datetime as dt 
import pandas as pd 
from tqdm import tqdm 

import astropy 
import astropy.units as u 

import astroplan 
import pytz 
import timezonefinder as tzf 
from astroquery.simbad import Simbad















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















# Create a Target (either by name, such as "Vega" or "Zeta Tau", or by ra/dec position) 
def get_target(target_name=None, target_radec_str=None): 

    """
    Create a SkyCoord target from either a name or RA/Dec string.

    Parameters
    ----------
    target_name : str, optional
        Name of the object (e.g., "Vega", "Zeta Tau") 
        Uses SIMBAD to look up coordinates 

    target_radec_str : str, optional 
        RA/Dec in HMS/DMS format:
        "HH MM SS.ss ±DD MM SS.ss"

    Returns
    -------
    astropy.coordinates.SkyCoord
        Target coordinates 

    Examples
    --------
    >>> get_target(target_name="Vega")
    >>> get_target(target_radec_str="18 36 56.34 +38 47 01.28")
    """

    if (target_name is None and target_radec_str is None) or (target_name is not None and target_radec_str is not None): 
        raise ValueError("Must provide either 'target_name' or 'target_radec_str' (but not both)")

    # Look up Ra/dec of target by name using Simbad 
    if target_name is not None: 
        result = Simbad.query_object(target_name)
        
        if result is None:
            raise ValueError(f"Object '{target_name}' not found in SIMBAD")
    
        ra_str = result['ra'][0]   
        dec_str = result['dec'][0] 
        coord = astropy.coordinates.SkyCoord(ra_str, dec_str, unit=(u.deg, u.deg))
        target_radec_str = coord.to_string('hmsdms', sep=' ')

    Target = astropy.coordinates.SkyCoord(
        target_radec_str,
        unit=(u.hourangle, u.deg)
    )

    # Include target name in Target object (i.e., "Vega" or "Zeta Tau")
    Target.name = target_name 

    # Include target ra/dec string for the plot title 
    target_coord_str = Target.to_string('hmsdms', precision=1) 
    target_coord_str = target_coord_str.replace("h", ":")
    target_coord_str = target_coord_str.replace("m", ":", 1)
    target_coord_str = target_coord_str.replace("s", "", 1)
    target_coord_str = target_coord_str.replace("d", "°")
    target_coord_str = target_coord_str.replace("m", "\'")
    target_coord_str = target_coord_str.replace("s", "\"")
    Target.coord_str = target_coord_str

    return Target 















# Calculate altitude of the Sun and the Target for a range of times 
def calc_visibility(Observer, Target, num_yrs: int = 2, spacing_minutes=10): 
    """
    Compute Sun and target altitudes over a time grid for visibility analysis.

    This function generates a regularly sampled time grid spanning one or more years,
    and computes the altitude of both the Sun and a target at each time. The results
    are suitable for visualization (e.g., heatmaps of observability, twilight regions).

    Parameters
    ----------
    Observer : astroplan.Observer
        Observer object defining the location and timezone. 
        Typically created with `get_observer()`. 
        
    Target : astropy.coordinates.SkyCoord
        Target coordinates. Typically created with `get_target()`.

    num_yrs : int, optional
        Number of years to compute starting from Dec 31, 2025 (default = 2).
        - 1 → one-year span
        - 2 → two-year span (useful to avoid edge effects across Dec-Jan boundary) 

    spacing_minutes : int, optional
        Time resolution of the grid in minutes (default = 10).
        Smaller values give higher precision but increase computation time. 

    Returns
    -------
    dates : pandas.DatetimeIndex
        Array of dates (one per day, anchored at local noon).
    times : array-like of datetime
        Time axis for one day (local time), used as the vertical axis in plots.
    target_alt : 2D numpy.ndarray
        Target altitude in degrees, shape (time, date).
    sun_alt : 2D numpy.ndarray
        Sun altitude in degrees, shape (time, date). 
        This allows you to calculate the visibility without caring about the night/day cutoff 
        Later, you can mask 'target_alt' based on sun_alt being above any threshold you like 
        (0 degrees for civil twilight, -18 degrees to require full night, or any value in between, such as -14.548)

    sun_alt : 2D numpy.ndarray
        Sun altitude in degrees, shape (time, date).

        This is precomputed so that visibility constraints can be applied
        flexibly at the plotting or analysis stage, rather than being fixed
        during calculation.

        The sun altitude array can be used to define arbitrary observing
        conditions after the fact, for example:

        - sun_alt < 0°   → only requires it to be past sunset 
        - sun_alt < -6°  → stricter condition: civil twilight is not allowed (nautical, astronomical, full night allowed)
        - sun_alt < -12° → only astronomical twilight and full night allowed 
        - sun_alt < -18° → only full night allowed 
        - sun_alt < 14.348234° → you can set the cutoff to be anything you want 

        This design allows you to compute the visibility once and reuse it for different observing criteria 

    Notes
    -----
    - The time grid is constructed from local noon to noon (not midnight), which
      aligns better with astronomical observing nights.
    - Using two years helps prevent features from being split across year boundaries,
      similar to plotting multiple periods in a phase-folded light curve.

    Examples
    --------
    >>> obs = get_observer(observatory_name="Winer")
    >>> target = get_target(target_name="Vega")
    >>> dates, times, target_alt, sun_alt = calc_visibility(obs, target)

    >>> # Higher resolution (slower)
    >>> calc_visibility(obs, target, spacing_minutes=5)

    >>> # One-year view
    >>> calc_visibility(obs, target, num_yrs=1)
    """

    print("Calculating visibility...") 
    print(f"Target: {Target.name} ({Target.coord_str})")
    print(f"Observer: {Observer.name} ({Observer.coord_str})") 
    print(f"Number of years: {num_yrs}") 
    print(f"Minutes between each data point (resolution): {spacing_minutes}")

    # Dates: Every day from December 31st until num_yrs (1 or 2) years ahead 
    dates = pd.date_range(
        start='2025-12-31 12:00:00',
        end=f'{2025+num_yrs}-12-31 12:00:00',
        freq='1D', # 3D = Calculate visibility every 3rd day, 10D = every 10th day, etc 
        tz=str(Observer.timezone)
    )

    # Samples: how many points per day to calculate the alitudes 
    samples_per_day = int(24 * 60 / spacing_minutes)  

    # Time relative to noon each day (0 → 24h) (1d continuous array)
    offsets = astropy.time.TimeDelta(np.linspace(0, 24*3600, samples_per_day), format='sec') 

    # Create 2d grid by folding the 1d continuous array so that each column corresponds to one day 
    date_astroplan = astropy.time.Time(dates.tz_convert('UTC').to_pydatetime())
    time_grid = date_astroplan[:, None] + offsets[None, :]
    times_flat = time_grid.flatten()

    # Y axis variable on plots 
    time_grid_dt = time_grid.to_datetime() 
    times = time_grid_dt[0]

    # Build AltAz frame (used by both Target and Sun calculation)
    altaz_frame = astropy.coordinates.AltAz(obstime=times_flat, location=Observer.location)

    # Calculate target altitudes 
    print("Calculating target altitudes...")
    target_alt = Target.transform_to(altaz_frame).alt.to_value()
    target_alt = target_alt.reshape(time_grid.shape) # Fold 1d array back into the 2d array 
    target_alt = target_alt.T

    # Calculate sun altitudes 
    print("Calculating Sun altitudes...")
    sun_alt = astropy.coordinates.get_sun(times_flat).transform_to(altaz_frame).alt.to_value()
    sun_alt = sun_alt.reshape(time_grid.shape) # Fold 1d array back into the 2d array 
    sun_alt = sun_alt.T # Transpose to make sure it treats X as dates and Y as times 
    print("Done. \n")

    return (
        dates,          # 1d array of dates (x axis)
        times,          # 1d array of times (y axis)
        target_alt,     # 2d array of target altitude at every date and time 
        sun_alt         # 2d array of Sun altitude at every date and time 
    ) 















def plot_visibility(
        dates, 
        times, 
        target_alt, 
        sun_alt, 
        Target, 
        Observer, 
        target_min_alt=25, 
        sun_max_alt=-6
    ): 
    """
    Plot a 2D visibility map showing when a target is observable from a given site.

    This function visualizes:
    - Solar twilight regions (background shading)
    - Target visibility above a minimum altitude (binary overlay)
    - Target altitude as a continuous colormap during observable conditions

    The resulting plot shows date on the x-axis and local time on the y-axis,
    providing an intuitive “observability map” for planning observations.

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        Array of dates (one per column in the grid). 

    times : array-like of datetime
        Time axis for one day (local time), used as the y-axis.

    target_alt : 2D numpy.ndarray
        Target altitude in degrees, shape (time, date). 

    sun_alt : 2D numpy.ndarray
        Sun altitude in degrees, shape (time, date).

    Target : astropy.coordinates.SkyCoord
        Target object. Expected to have attributes like `.name` and `.coord_str`. 

    Observer : astroplan.Observer
        Observer object. Expected to have attributes like `.name`, `.coord_str`,
        and `.timezone`.

    target_min_alt : float, optional
        Minimum altitude (in degrees) required for the target to be 
        considered observable (default = 25). 

    sun_max_alt : float, optional
        Maximum Sun altitude (in degrees) for acceptable observing conditions.
        Default = -6 (civil twilight limit). Use -12 or -18 for darker conditions.

    Returns
    -------
    None
        Displays a matplotlib figure.

    Notes
    -----
    The plot consists of three layered components:

    1. **Twilight background**
        Colored bands indicate solar altitude ranges: 
        - Full night: Sun < -18°
        - Astronomical twilight: -18° ≤ Sun < -12°
        - Nautical twilight: -12° ≤ Sun < -6°
        - Civil twilight: -6° ≤ Sun < 0°
        - Civil daylight: 0° ≤ Sun < 6°
        - Nautical daylight: 6° ≤ Sun < 12°
        - Astronomical daylight: 12° ≤ Sun < 18°
        - Full day: Sun ≥ 18°

    2. **Target visibility mask**
       Semi-transparent overlay showing when the target is above `target_min_alt`.

    3. **Target altitude colormap**
       Continuous shading (grayscale) showing altitude, only where both:
       - target_alt > target_min_alt (the target is up)
       - sun_alt <= sun_max_alt (it's night) 

    Examples
    --------
    >>> dates, times, target_alt, sun_alt = calc_visibility(obs, target)
    >>> plot_visibility(dates, times, target_alt, sun_alt, target, obs)

    >>> # Stricter darkness requirement (astronomical night)
    >>> plot_visibility(dates, times, target_alt, sun_alt, target, obs,
    ...                 sun_max_alt=-18)

    >>> # Lower altitude cutoff
    >>> plot_visibility(dates, times, target_alt, sun_alt, target, obs,
    ...                 target_min_alt=20)
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
    ax.set_xlim(dates[0], dates[-1]) 
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
    ax.xaxis.set_major_formatter(format_date)

    # Y-axis: Local time  
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.timezone(str(Observer.timezone)))) 
    ax.set_ylabel(f"Local time ({str(Observer.timezone)})") 
    ax.yaxis.set_major_locator(mdates.HourLocator(interval=2)) 
    ax.yaxis.set_minor_locator(mdates.HourLocator(interval=1)) 

    # Title    
    ax.set_title(f"Observer: {Observer.name} ({Observer.coord_str}) \nTarget: {Target.name} ({Target.coord_str})")



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
    plt.contourf(
        dates,
        times,
        sun_alt, 
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
    plt.contour(
        dates, 
        times, 
        sun_alt, 
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
    plt.contourf(
        dates,
        times,
        target_alt,  
        levels=levels,
        colors=colors, 
    )

    # Border: 
    plt.contour(
        dates,
        times,
        target_alt,  
        levels=levels,
        colors="black", 
        linewidths=1, 
        zorder=5, 
    )



    # 3: Target altitude colormap 

    # Display altitude only where target is up and sun is down 
    visible = (target_alt > target_min_alt) & (sun_alt <= sun_max_alt)
    Z = np.ma.MaskedArray(target_alt, mask=~visible)
    dark_greys = mcolors.LinearSegmentedColormap.from_list("dark_greys",cm.Greys(np.linspace(0.3, 1, 256)))
    target_alt_plot = plt.pcolormesh(
        dates,
        times,
        Z, 
        cmap=dark_greys, 
        zorder=4, 
        vmin=0, 
        vmax=90, 
    )

    # Add red contour around visibile region to make it pop more 
    plt.contour(
        dates,
        times,
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













