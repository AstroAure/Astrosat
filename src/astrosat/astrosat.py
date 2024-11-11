import os
import glob
import requests
from tqdm import tqdm
import cv2
import ffmpeg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.dates as mdates
import pandas as pd
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
from poliastro.twobody.sampling import EpochsArray
from poliastro.earth.util import raan_from_ltan
from poliastro.core.propagation import func_twobody
from poliastro.core.perturbations import J2_perturbation
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from astral import LocationInfo
from astral.sun import sun
import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, CartesianRepresentation, CartesianDifferential, TEME, ITRS, ICRS, EarthLocation, AltAz
from astropy.visualization import time_support
time_support(format='isot', scale='utc')
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

import warnings
warnings.filterwarnings("ignore", message=".*ERFA.*")


### UTILS ###

def units_arange(start:u.Quantity, end:u.Quantity, step:u.Quantity):
    """
    Generate an array of values with units, starting from 'start' to 'end' with a given 'step'.

    Parameters:
    start (u.Quantity): The starting value with units.
    end (u.Quantity): The ending value with units.
    step (u.Quantity): The step size with units.

    Returns:
    u.Quantity: An array of values with the same units as the input.
    """
    end = end.to(start.unit)
    step = step.to(start.unit)
    units_range = np.arange(start.value, end.value, step.value)
    return units_range*start.unit


### NASA JPL HORIZONS API ###

def request_horizons(**kwargs):
    """
    Sends a request to the NASA JPL Horizons system with the given parameters.

    This function writes the provided keyword arguments to a temporary file,
    sends the file to the Horizons API, and then deletes the temporary file.
    The response from the API is returned as a string.

    Parameters:
    **kwargs: Arbitrary keyword arguments to be included in the Horizons request.
    (https://ssd-api.jpl.nasa.gov/doc/horizons.html)

    Returns:
    str: The response text from the Horizons API.

    Example:
    response = request_horizons(COMMAND='10', CENTER='500@10', MAKE_EPHEM='YES')
    """
    with open("horizon_request.txt", 'w') as f:
        f.write("!$$SOF"+'\n')
        for arg in kwargs:
            f.write(f"{arg}='{kwargs[arg]}'"+'\n')
    with open("horizon_request.txt", "r") as f:
        url = 'https://ssd.jpl.nasa.gov/api/horizons_file.api'
        r = requests.post(url, data={'format':'text'}, files={'input': f})
    os.remove("horizon_request.txt")
    return r.text

def horizons_to_dataframe(text):
    """
    Converts a text block from the Horizons system into a pandas DataFrame.

    The function parses the text to extract data between the "$$SOE" (Start of Ephemeris) 
    and "$$EOE" (End of Ephemeris) markers. It assumes that the data is in CSV format 
    with a header row three lines above the "$$SOE" marker.

    Parameters:
    text (str): The input text containing the Horizons data.

    Returns:
    pd.DataFrame: A DataFrame containing the parsed data with appropriate column names.
    """
    def parse_str(string):
        try:
            return float(string.strip())
        except:
            return string.strip()
    lines = text.splitlines()
    start = lines.index("$$SOE")+1
    end = lines.index("$$EOE")
    names = [n.strip() for n in lines[start-3].split(",")[:-1]]
    data = [[parse_str(e) for e in l.split(",")[:-1]] for l in lines[start:end]]
    df = pd.DataFrame(data=data, columns=names)
    return df

def horizons_to_TEME(df, time='JDUT'):
    """
    Convert a DataFrame containing position and velocity data from the HORIZONS system to the TEME frame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the following columns:
        - 'X': X position in kilometers
        - 'Y': Y position in kilometers
        - 'Z': Z position in kilometers
        - 'VX': X velocity in kilometers per second
        - 'VY': Y velocity in kilometers per second
        - 'VZ': Z velocity in kilometers per second
        - time (str): Column name for the time data in Julian Date (default is 'JDUT')

    Returns:
    astropy.coordinates.builtin_frames.teme.TEME: TEME frame object containing the position and velocity data.
    """
    epoch = Time(df[time], scale='utc', format='jd')
    pos = CartesianRepresentation(df['X'], df['Y'], df['Z'], unit=u.km)
    vel = CartesianDifferential(df['VX'], df['VY'], df['VZ'], unit=u.km/u.s)
    teme = TEME(pos.with_differentials(vel), obstime=epoch)
    return teme


### PLOT ####

def plot_world(color=False, figsize=(10,6), projection=ccrs.PlateCarree(), resolution='low'):
    """
    Plots a world map using Cartopy.

    Parameters:
    color (bool): If True, uses a stock image for the map. If False, uses coastlines.
    figsize (tuple): Size of the figure in inches, default is (10, 6).
    projection (ccrs.Projection): Cartopy projection to use for the map, default is PlateCarree.
    resolution (str): Resolution of the coastlines. Options are 'low', 'mid', 'high'. Default is 'low'.

    Returns:
    fig (matplotlib.figure.Figure): The created figure.
    ax (matplotlib.axes._subplots.AxesSubplot): The created subplot axes.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection':projection})
    if color: 
        ax.stock_img()
    else: 
        ax.set_global()
        res_dict = {'low':'110m', 'mid':'50m', 'high':'10m'}
        ax.coastlines(resolution=res_dict[resolution])
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    return fig, ax

def plot_trace(teme, ax=None, color='plasma'):
    """
    Plots the trace of a satellite's trajectory on a world map.

    Parameters:
    teme : `~astropy.coordinates.BaseCoordinateFrame`
        The satellite's position in TEME (True Equator Mean Equinox) coordinates.
    ax : `matplotlib.axes._subplots.AxesSubplot`, optional
        The matplotlib axes to plot on. If None, a new world map will be created.
    color : str, optional
        The colormap to use for the trace. Default is 'plasma'. Must be a valid colormap name.

    Returns:
    ax : `matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes with the plotted trace.
    """
    if ax is None:ax = plot_world()[1]
    geo = teme.transform_to(ITRS(obstime=teme.obstime)).earth_location.geodetic
    if color in list(colormaps):
        cmap = colormaps.get_cmap(color)
        color = cmap(np.linspace(0,1,len(geo.lon)))
    ax.scatter(geo.lon, geo.lat, transform=ccrs.Geodetic(), marker='o', c=color, s=1)
    return ax

def plot_earth_3D(ax, res=20):
    """
    Plots a 3D wireframe representation of the Earth on the given Axes object.

    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axes on which to plot the Earth.
    res (int, optional): The resolution of the wireframe grid. Higher values result in a finer grid. Default is 20.

    Returns:
    None
    """
    R = Earth.R.to(u.km).value
    u_e, v_e = np.mgrid[0:2*np.pi:res*1.j, 0:np.pi:res*1.j]
    x = R*np.cos(u_e)*np.sin(v_e)
    y = R*np.sin(u_e)*np.sin(v_e)
    z = R*np.cos(v_e)
    ax.plot_wireframe(x, y, z, color="b", alpha=0.1, label='Earth', zorder=-1)

def plot_3D_cartesian(cartesian_dict, figsize=(15,8), lim=3e4):
    """
    Plots 3D Cartesian coordinates in three orthogonal projections (XY, XZ, YZ planes).

    Parameters:
    cartesian_dict (list of dict): A list of dictionaries where each dictionary represents an object with the following keys:
        - 'cartesian': A `~astropy.coordinates.CartesianRepresentation` object representing the object's position.
        - 'color' (optional): A string representing the color of the plot.
        - 'label' (optional): A string representing the label of the object.
    figsize (tuple, optional): A tuple representing the size of the figure. Default is (15, 8).
    lim (float, optional): The limit for the axes. Default is 3e4.

    Returns:
    None
    """
    fig, axs = plt.subplots(1,3,figsize=(15,8), subplot_kw={'projection':'3d'}, gridspec_kw={'hspace':0, 'wspace':0})
    for ax in axs:
        for obj in cartesian_dict:
            pos = obj['cartesian']
            color = obj['color'] if 'color' in obj else None
            label = obj['label'] if 'label' in obj else None
            ax.plot(pos.x, pos.y, pos.z, color=color, label=label, zorder=3)
        plot_earth_3D(ax)
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_zlim(-lim,lim)
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_proj_type('ortho')
    axs[0].view_init(elev=90, azim=-90)
    axs[0].set_title("XY plane")
    axs[1].view_init(elev=0, azim=-90)
    axs[1].set_title("XZ plane")
    axs[2].view_init(elev=0, azim=0)
    axs[2].set_title("YZ plane")
    axs[0].legend(loc='lower left')
    plt.show()

def dawn_dusk(start, end, lat, lon, darkness='civil'):
    """
    Calculate the dawn and dusk times for a given location and date range.

    Parameters:
    start (datetime or astropy.time.Time): The start date and time.
    end (datetime or astropy.time.Time): The end date and time.
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    darkness (str, optional): Type of twilight to consider ('civil', 'nautical', 'astronomical'). 
                              Defaults to 'civil'.

    Returns:
    list: A list of lists, where each inner list contains two elements:
          [dusk_time, dawn_time] for each night in the date range.
          If dusk or dawn time is not available, it will be None.
    """
    if type(start)==Time: start=start.to_datetime()
    if type(end)==Time: end=end.to_datetime()
    if start.hour < 12: start -= datetime.timedelta(days=1)
    if end.hour > 12: end += datetime.timedelta(days=1)
    start = start.date()
    end = end.date()
    loc = LocationInfo("", "", "", lat, lon)
    dark_dict = {'civil':6.0, 'nautical':12.0, 'astronomical':18.0}
    date_range = pd.date_range(start, end=end, freq='1d')
    nights = []
    for i in range(len(date_range)):
        date = date_range[i]
        try:
            sun_info = sun(loc.observer, date=date, dawn_dusk_depression=dark_dict[darkness], tzinfo='utc')
        except:
            sun_info = {'dusk':None, 'dawn':None}
        if i != len(date_range)-1:
            nights.append([sun_info['dusk'], None])
        if i != 0:
            nights[i-1][1] = sun_info['dawn']
    return nights

def plot_night(ax, start, end, lat, lon, darkness=['civil', 'nautical', 'astronomical'], alpha=0.2):
    """
    Plots periods of night on a given matplotlib axis.

    Parameters:
    ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
    start (datetime.datetime or astropy.time.Time): The start time of the plotting range.
    end (datetime.datetime or astropy.time.Time): The end time of the plotting range.
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    darkness (list of str, optional): Types of darkness to plot. Options are 'civil', 'nautical', and 'astronomical'. Defaults to ['civil', 'nautical', 'astronomical'].
    alpha (float, optional): Transparency level of the night shading. Defaults to 0.2.

    Returns:
    None
    """
    if type(start)==Time: start=start.to_datetime()
    if type(end)==Time: end=end.to_datetime()
    for dark in darkness:
        nights = dawn_dusk(start, end, lat, lon, darkness=dark)
        for night in nights:
            if (night[0] is not None) and (night[1] is not None):
                ax.axvspan(night[0], night[1], facecolor='k', edgecolor='none', alpha=alpha)
    ax.set_xlim(start,end)

def plot_atmosphere(ax, start, end):
    """
    Plots the atmosphere on a given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to plot the atmosphere.
    start (float): The starting x-coordinate for the plot.
    end (float): The ending x-coordinate for the plot.

    Returns:
    None
    """
    ax.imshow([[0,0],[0.5,0.5]], 
              cmap=colormaps.get_cmap('Blues'), 
              interpolation='bicubic',
              extent=[start, end, Earth.R.to(u.km).value, Earth.R.to(u.km).value+100], 
              aspect='auto', vmin=0, vmax=1)


### ORBIT ###

def sso_inclination(a):
    """
    Calculate the inclination for a Sun-synchronous orbit (SSO).

    Parameters:
    a (Quantity): Semi-major axis of the orbit, with units of length.

    Returns:
    Quantity: Inclination of the orbit in degrees.
    """
    return np.arccos(-(a/(12352*u.km))**3.5).to(u.deg)

def orbit_to_TEME(orbit, epoch, perturbations=func_twobody):
    """
    Converts an orbit to the True Equator Mean Equinox (TEME) coordinate system.

    Parameters:
    orbit (Orbit): The orbit object to be converted.
    epoch (datetime): The epoch at which the conversion is to be performed.
    perturbations (function, optional): The perturbation function to be used in the conversion. 
                                        Defaults to `~poliastro.core.propagation.func_twobody`.

    Returns:
    TEME: The position in the TEME coordinate system.
    """
    ephem = orbit.to_ephem(strategy=EpochsArray(epochs=epoch, method=CowellPropagator(f=perturbations)))
    pos = ephem.sample(ephem.epochs)
    teme = TEME(pos, obstime=epoch)
    return teme

def J2(t0, state, k):
    """
    Compute the time derivative of the state vector considering both the 
    two-body problem and the J2 perturbation.

    Parameters:
    t0 : float
        The initial time.
    state : array_like
        The state vector of the satellite [x, y, z, vx, vy, vz].
    k : float
        The gravitational parameter of the central body.

    Returns:
    numpy.ndarray
        The time derivative of the state vector including the J2 perturbation.
    """
    du_kep = func_twobody(t0, state, k)
    ax, ay, az = J2_perturbation(
        t0, state, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
    )
    du_ad = np.array([0, 0, 0, ax, ay, az])
    return du_kep + du_ad


### ASTRONOMY ###

def cartesian_to_radec(cartesian):
    """
    Convert Cartesian coordinates to Right Ascension (RA) and Declination (Dec).

    Parameters:
    cartesian (CartesianRepresentation): A Cartesian representation of the coordinates.

    Returns:
    SkyCoord: The corresponding coordinates in the ICRS frame with RA and Dec in degrees.
    """
    ra = np.arctan2(cartesian.y, cartesian.x).to(u.deg)
    ra = np.putmask(ra, ra<0 ,ra+360*u.deg)
    dec = np.arcsin(cartesian.z/cartesian.norm()).to(u.deg)
    coord = SkyCoord(ra, dec, frame='icrs')
    return coord

def astrometry_target_plot(observer_teme, target_teme, mask=None, 
                           fig=None, ax=None, 
                           figsize=(10,5), color='plasma',
                           target_name='target',
                           return_radec=False,
                           return_color=False):
    """
    Plots the astrometric position of a target object as observed from a satellite.

    Parameters:
    observer_teme : `~astropy.coordinates.BaseCoordinateFrame`
        The observer's position in TEME (True Equator Mean Equinox) frame.
    target_teme : `~astropy.coordinates.BaseCoordinateFrame`
        The target's position in TEME frame.
    mask : array-like, optional
        A boolean mask to filter the target positions.
    fig : `~matplotlib.figure.Figure`, optional
        The figure object to plot on. If None, a new figure is created.
    ax : `~matplotlib.axes.Axes`, optional
        The axes object to plot on. If None, new axes are created.
    figsize : tuple, optional
        The size of the figure, default is (10, 5).
    color : str, optional
        The colormap to use for the scatter plot, default is 'plasma'.
    target_name : str, optional
        The name of the target object, default is 'target'.
    return_radec : bool, optional
        If True, returns the right ascension and declination of the target.
    return_color : bool, optional
        If True, returns the color array used in the scatter plot.

    Returns:
    --------
    target_obs_radec : `~astropy.coordinates.SkyCoord`
        The right ascension and declination of the target if `return_radec` is True.
    color : array-like
        The color array used in the scatter plot if `return_color` is True.

    Notes:
    ------
    The function calculates the relative position of the target with respect to the observer,
    converts it to right ascension and declination, and plots it on a scatter plot.
    """
    target_obs = target_teme.cartesian.without_differentials() - observer_teme.cartesian.without_differentials()
    target_obs_radec = cartesian_to_radec(target_obs)
    n = len(target_obs_radec.ra)
    if mask is not None: target_obs_radec = target_obs_radec[mask]
    new_fig = (fig is None) or (ax is None)
    if new_fig: fig, ax = plt.subplots(figsize=figsize)
    if color in list(colormaps):
        cmap = colormaps.get_cmap(color)
        color = cmap(np.linspace(0,1,n))
        if mask is not None: color = color[mask]
    ax.scatter(target_obs_radec.ra, target_obs_radec.dec, marker='o', c=color, s=1)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Right ascension (°)')
    ax.set_ylabel('Declination (°)')
    ax.set_title(f"Astronomical position of {target_name} from satellite")
    if new_fig: fig.tight_layout()
    if new_fig: plt.show()
    if return_radec and not return_color: return target_obs_radec
    if return_radec and return_color: return target_obs_radec, color

def proper_motion(radec, dt=None, epoch=None):
    """
    Calculate the proper motion of celestial objects.

    Parameters:
    radec : `astropy.coordinates.SkyCoord`
        The right ascension and declination coordinates of the celestial objects.
    dt : `astropy.units.Quantity`, optional
        The time interval over which to calculate the proper motion. If not provided, `epoch` must be specified.
    epoch : tuple of `astropy.time.Time`, optional
        A tuple containing two `Time` objects representing the start and end epochs. If provided, `dt` will be calculated as the difference between these epochs.

    Returns:
    pm : `astropy.units.Quantity`
        The proper motion in arcseconds per second.

    Raises:
    ValueError
        If neither `dt` nor `epoch` is specified.
    """
    if dt is None and epoch is None: raise ValueError("dt or epoch must be specified (if both, only dt will be used) !")
    if dt is None:
        dt = (epoch[1]-epoch[0]).to_value(unit='s')*u.s
    pm = radec.separation(np.roll(radec, 1))[1:].to(u.arcsec)/dt
    return pm

def earth_target_eclipse(observer_teme, target_teme):
    """
    Calculate the closest distance between the observer and the target, 
    ensuring that the distance is not less than the Earth's radius.

    Parameters:
    observer_teme : astropy.coordinates.BaseCoordinateFrame
        The position of the observer in the TEME (True Equator Mean Equinox) frame.
    target_teme : astropy.coordinates.BaseCoordinateFrame
        The position of the target in the TEME frame.

    Returns:
    numpy.ndarray
        An array of distances from the observer to the target, with values 
        less than the Earth's radius set to the Earth's radius.
    """
    pos = observer_teme.cartesian.xyz.T
    dir = target_teme.cartesian.xyz.T - pos
    closest_t = -np.sum(pos*dir, axis=1)/np.linalg.norm(dir,axis=1)**2
    closest_t[closest_t<0] = 0
    closest_earth = np.linalg.norm(pos + dir*np.tile(closest_t, (3,1)).T, axis=1)
    closest_earth[closest_earth<=Earth.R] = Earth.R
    return closest_earth

def target_visibility_plot(observer_teme, target_teme, 
                           fig=None, ax=None, 
                           figsize=(10,5), color='k', 
                           target_name='target', 
                           return_vis=False):
    """
    Plots the visibility of a target from a satellite observer.

    Parameters:
    observer_teme : `astropy.coordinates.SkyCoord`
        The satellite observer's position in TEME coordinates.
    target_teme : `astropy.coordinates.SkyCoord`
        The target's position in TEME coordinates.
    fig : `matplotlib.figure.Figure`, optional
        The figure object to plot on. If None, a new figure is created.
    ax : `matplotlib.axes.Axes`, optional
        The axes object to plot on. If None, new axes are created.
    figsize : tuple, optional
        The size of the figure in inches. Default is (10, 5).
    color : str, optional
        The color of the plot. Default is 'k' (black).
    target_name : str, optional
        The name of the target to be displayed in the plot. Default is 'target'.
    return_vis : bool, optional
        If True, the function returns the visibility data. Default is False.

    Returns:
    closest_earth : `astropy.units.Quantity`, optional
        The distance between the target and the Earth center, if `return_vis` is True.

    Notes:
    The function plots the distance between the target and the Earth center over time,
    highlighting the Earth's radius and the atmospheric layers.
    """
    closest_earth = earth_target_eclipse(observer_teme, target_teme)
    time = observer_teme.obstime.to_datetime()
    new_fig = (fig is None) or (ax is None)
    if new_fig: fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time, closest_earth, c=color)
    ax.fill_between(time, closest_earth, Earth.R.to(u.km).value, color=color, alpha=0.15)
    ax.axhline(Earth.R.to(u.km).value, c='r', label='Earth radius')
    plot_atmosphere(ax, time[0], time[-1])
    ax.set_ylim(Earth.R.to(u.km).value-100, np.max(closest_earth).to(u.km).value+100)
    ax.set_ylabel(f"{target_name.capitalize()}-Satellite distance\nto Earth center (km)")
    ax.set_title(f"{target_name.capitalize()} visibility from satellite")
    ax.legend(loc='lower right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig.autofmt_xdate(rotation=20)
    if new_fig: fig.tight_layout()
    if new_fig: plt.show()
    if return_vis: return closest_earth

def TEME_to_altaz(teme, loc):
    """
    Convert TEME (True Equator Mean Equinox) coordinates to AltAz (Altitude-Azimuth) coordinates.

    Parameters:
    teme : `astropy.coordinates.SkyCoord`
        The input coordinates in the TEME frame.
    loc : `astropy.coordinates.EarthLocation`
        The observer's location on Earth.

    Returns:
    altaz : `astropy.coordinates.SkyCoord`
        The coordinates in the AltAz frame.
    """
    itrs = teme.transform_to(ITRS(obstime=teme.obstime))
    itrs_repr = itrs.cartesian.without_differentials() - loc.get_itrs(teme.obstime).cartesian
    itrs_topo = ITRS(itrs_repr, obstime=teme.obstime, location=loc)
    altaz = itrs_topo.transform_to(AltAz(obstime=teme.obstime, location=loc))
    return altaz

def airmass_plot(teme, loc, fig=None, ax=None, color='plasma', figsize=(10,5), ylim=(0,90)):
    altaz = TEME_to_altaz(teme, loc)
    time = teme.obstime.to_datetime()
    new_fig = (fig is None) or (ax is None)
    if new_fig: fig, ax = plt.subplots(figsize=figsize)
    plot_night(ax, time[0], time[-1], loc.lat.value, loc.lon.value)
    if color in list(colormaps):
        cmap = colormaps.get_cmap(color)
        color = cmap(np.linspace(0,1,len(time)))
    ax.scatter(time, altaz.alt, marker='o', c=color, s=1)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel("Altitude (°)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig.autofmt_xdate(rotation=20)
    if new_fig: fig.tight_layout()
    if new_fig: plt.show()

def UVW_coordinate_frame(teme):
    """
    Calculate the UVW coordinate frame from a given TEME (True Equator Mean Equinox) frame.
    Parameters:
    teme : astropy.coordinates.BaseCoordinateFrame
        The TEME frame containing the position and velocity vectors.
    Returns:
    numpy.ndarray
        A 3x3 numpy array where each row represents the U, V, and W unit vectors respectively.
        - U: Unit vector in the direction of the position vector.
        - V: Unit vector perpendicular to U and W, forming a right-handed coordinate system.
        - W: Unit vector in the direction of the angular momentum vector.
    """
    r = teme.cartesian.xyz.to(u.km).value
    v = teme.cartesian.differentials['s'].d_xyz.to(u.km/u.s).value
    h = np.cross(r, v, axis=0)

    W = h/np.linalg.norm(h, axis=0)
    U = r/np.linalg.norm(r, axis=0)
    V = -np.cross(U, W, axis=0) # https://ai-solutions.com/_help_Files/attitude_reference_frames.htm

    return np.array([U,V,W])

def calc_altaz_error(uvw_error, teme, gs_loc): 
    """
    Calculate the altitude-azimuth error for a satellite given its position error in the UVW coordinate frame.

    Parameters:
    uvw_error (array-like): The position error in the UVW coordinate frame.
    teme (TEME): The satellite's position in the TEME coordinate frame.
    gs_loc (EarthLocation): The ground station location.

    Returns:
    AltAz: The altitude-azimuth coordinates of the satellite with the position error applied.
    """
    sat_uvw = UVW_coordinate_frame(teme)
    pos_error = sat_uvw[0]*uvw_error[0] + sat_uvw[1]*uvw_error[1] + sat_uvw[2]*uvw_error[2]
    error_sat_teme = TEME(teme.cartesian.xyz + pos_error, obstime=teme.obstime.to_datetime())
    error_sat_altaz = TEME_to_altaz(error_sat_teme, gs_loc)
    return error_sat_altaz


### IMAGE SIMULATION ###

star_catalogs = {'UCAC4' : {'id':'I/322A/out', 'mag':'Vmag', 'ra':'RAJ2000', 'dec':'DEJ2000'},
                 'GaiaDR3' : {'id':'I/355/gaiadr3', 'mag':'Gmag', 'ra':'RA_ICRS', 'dec':'DE_ICRS'},}

def mag_to_marker_size(mag, limiting_mag=10.0):
    """
    Convert magnitude to marker size for plotting.

    Parameters:
    mag (array-like): Array of magnitudes.
    limiting_mag (float, optional): The limiting magnitude. Default is 10.0.

    Returns:
    array-like: Array of marker sizes corresponding to the input magnitudes.
                Marker size is calculated as (0.5 + limiting_mag - mag)**2.
                If a magnitude is greater than the limiting magnitude, the marker size is set to 0.
    """
    size = (0.5 + limiting_mag - mag)**2
    size[mag>limiting_mag] = 0
    return size

def str_fov(field_fov):
    """
    Convert a list of field of view (FOV) values to a formatted string.

    Parameters:
    field_fov (list of astropy.units.Quantity): A list of FOV values with units (e.g., degrees, arcminutes, arcseconds).

    Returns:
    str: A formatted string representing the FOV values, separated by ' x ' and with appropriate unit symbols.
    """
    unit = {u.deg:'°', u.arcmin:"'", u.arcsec:'"'}
    string = ' x '.join([f"{fov.value:.1f}{unit[fov.unit]}" for fov in field_fov])
    return string

def str_coord(coord):
    """
    Convert a coordinate object to a formatted string representation.

    Parameters:
    coord (object): An object with 'ra' (right ascension) and 'dec' (declination) attributes.
                    'ra' should have 'hms' (hours, minutes, seconds) attributes.
                    'dec' should have 'signed_dms' (degrees, minutes, seconds) attributes.

    Returns:
    str: A string representation of the coordinates in the format "HHhMMmSS.SS +DDD°MM'SS.SS".
    """
    ra = coord.ra.hms
    dec = coord.dec.signed_dms
    str_ra = f"{ra.h:0>2.0f}h{ra.m:0>2.0f}m{ra.s:0>5.2f}s"
    str_dec = f"{'+' if dec.sign==1 else '-'}{dec.d:0>3.0f}°{dec.m:0>2.0f}'{dec.s:0>5.2f}"+'"'
    return f"{str_ra} {str_dec}"

def plot_field(center, field_fov,
               catalog='UCAC4', limiting_mag=12.0,
               epoch=None,
               legend=True,
               fig=None, ax=None, figsize=(8,8),
               invert=True, no_stars=False):
    """
    Plots a star field centered on a given coordinate.

    Parameters:
    center (SkyCoord): The central coordinate of the field.
    field_fov (tuple): The field of view dimensions (width, height) in degrees.
    catalog (str, optional): The star catalog to use. Default is 'UCAC4'. Possible values are 'UCAC4' and 'GaiaDR3'.
    limiting_mag (float, optional): The limiting magnitude for stars to be plotted. Default is 12.0.
    epoch (Time, optional): The epoch of the observation. Default is None.
    legend (bool, optional): Whether to display the legend. Default is True.
    fig (Figure, optional): Matplotlib figure object. Default is None.
    ax (Axes, optional): Matplotlib axes object. Default is None.
    figsize (tuple, optional): Size of the figure. Default is (8, 8).
    invert (bool, optional): Whether to invert the colors. Default is True.
    no_stars (bool, optional): Whether to plot stars. Default is False.

    Returns:
    None
    """
    new_fig = (fig is None) or (ax is None)
    if new_fig: fig, ax = plt.subplots(figsize=figsize)
    # Stars
    if not no_stars:
        stars = Vizier.query_region(center, width=field_fov[0], height=field_fov[1], 
                                catalog=star_catalogs[catalog]['id'], 
                                column_filters={star_catalogs[catalog]['mag']:f'<{limiting_mag}'}).copy()[0]
        src_color = 'k' if invert else 'w'
        star_size = mag_to_marker_size(stars[star_catalogs[catalog]['mag']], limiting_mag=limiting_mag)
        ax.scatter(stars[star_catalogs[catalog]['ra']]*u.deg, 
                   stars[star_catalogs[catalog]['dec']]*u.deg, 
                   s=star_size, c=src_color)
    # Frame
    ax.set_xlim((center.ra-0.5*field_fov[0]).to(u.deg).value,(center.ra+0.5*field_fov[0]).to(u.deg).value)
    ax.set_ylim((center.dec-0.5*field_fov[1]).to(u.deg).value,(center.dec+0.5*field_fov[1]).to(u.deg).value)
    # Data
    if legend:
        txt_color = 'r' if invert else 'lime'
        ax.text(0.98, 0.02, f"V-mag < {limiting_mag:.1f}", 
                ha='right', va='bottom', transform = ax.transAxes, 
                color=txt_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
        ax.text(0.98, 0.07, f"FOV : {str_fov(field_fov)}", 
                ha='right', va='bottom', transform = ax.transAxes, 
                color=txt_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
        ax.text(0.02, 0.02, str_coord(center), 
                    ha='left', va='bottom', transform = ax.transAxes, 
                    color=txt_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
        if epoch is not None:
            ax.text(0.02, 0.07, epoch.to_value('iso', subfmt='date_hm'), 
                    ha='left', va='bottom', transform = ax.transAxes, 
                    color=txt_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
    # Background
    bkg_color = 'w' if invert else 'k'
    ax.set_facecolor(bkg_color)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.add_artist(ax.patch)
    ax.patch.set_zorder(-1)
    fig.patch.set_visible(False)
    fig.tight_layout()
    if new_fig: plt.show()

def add_map(ax, target, radec, 
            target_color='r', radec_color='lime', legend_color='lime', 
            shape_ins=[0.3,0.3], pad_ins=0.03):
    """
    Adds an inset map to the given axis with specified target and RA/DEC coordinates.

    Parameters:
    ax (matplotlib.axes.Axes): The main axis to which the inset map will be added.
    target (SkyCoord): The target coordinates (RA, DEC) to be highlighted on the map.
    radec (SkyCoord): The RA/DEC coordinates to be plotted on the map.
    target_color (str, optional): Color for the target marker. Default is 'r'.
    radec_color (str, optional): Color for the RA/DEC markers. Default is 'lime'.
    legend_color (str, optional): Color for the legend text and grid. Default is 'lime'.
    shape_ins (list, optional): Shape of the inset map as [width, height]. Default is [0.3, 0.3].
    pad_ins (float, optional): Padding around the inset map. Default is 0.03.

    Returns:
    None
    """
    ins = ax.inset_axes([1-shape_ins[0]-pad_ins,1-shape_ins[1]-pad_ins,shape_ins[0],shape_ins[1]])
    ins.set_xlabel('RA (°)', color=legend_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
    ins.set_ylabel('DEC (°)', color=legend_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
    ins.patch.set_alpha(0.2) # Transparent background
    ins.grid(True, which='both', alpha=0.2, color=legend_color, zorder=0) # Colored grid
    ins.scatter(radec.ra, radec.dec, marker='o', c=radec_color, s=1, zorder=10)
    ins.scatter(target.ra, target.dec, c=target_color, zorder=20)
    ins.set_aspect('equal', adjustable='datalim')
    for spine in ins.spines: # Colored splines
        ins.spines[spine].set(color=legend_color, alpha=0.5)
    ins.tick_params(axis='x', which='both', colors=legend_color) # Colored ticks
    ins.tick_params(axis='y', which='both', colors=legend_color) # Colored ticks
    for tick in ins.get_xticklabels()+ins.get_yticklabels(): # Colored labels
        tick.set(color=legend_color, fontsize='large', fontweight='bold', fontfamily='monospace')

def simulate_image(center, target_radec=None,
                   field_fov=(5.0*u.deg, 5.0*u.deg), 
                   limiting_mag=12.0, epoch=None,
                   target_mag=None, 
                   visibility=True, invert=False,
                   center_list=None, visibility_list=None, color_list=None):
    """
    Simulates an astronomical image centered on a given coordinate.

    Parameters:
    center : SkyCoord
        The central coordinate of the field.
    target_radec : SkyCoord, optional
        The RA/Dec coordinates of the target object.
    field_fov : tuple of Quantity, optional
        The field of view of the image in degrees (default is (5.0*u.deg, 5.0*u.deg)).
    limiting_mag : float, optional
        The limiting magnitude for the stars in the field (default is 12.0).
    epoch : Time, optional
        The epoch of the observation.
    target_mag : float, optional
        The magnitude of the target object.
    visibility : bool, optional
        Whether to display stars in the field (default is True).
    invert : bool, optional
        Whether to invert the colors of the plot (default is False).
    center_list : list of SkyCoord, optional
        List of coordinates to be marked on the map.
    visibility_list : list of bool, optional
        List indicating the visibility of each coordinate in center_list.
    color_list : list of str, optional
        List of colors for each coordinate in center_list.

    Returns:
    fig : Figure
        The matplotlib figure object.
    ax : Axes
        The matplotlib axes object.
    """
    fig, ax = plt.subplots(figsize=(8,8))
    # Stars
    plot_field(center, field_fov, limiting_mag=limiting_mag, epoch=epoch,
                        fig=fig, ax=ax, invert=invert, no_stars=not(visibility))
    # Target
    if target_radec is not None:
        if target_mag is not None:
            txt_color = 'r' if invert else 'lime'
            y_pos = 0.07 if epoch is None else 0.12
            ax.text(0.02, y_pos, f"Target V-mag : {target_mag:.1f}", 
                        ha='left', va='bottom', transform = ax.transAxes, 
                        color=txt_color, fontsize='xx-large', fontweight='bold', fontfamily='monospace')
        if visibility:
            if target_mag is None: target_mag = limiting_mag-8
            target_size = mag_to_marker_size(np.array([target_mag]), limiting_mag=limiting_mag)[0]
            ax.scatter(target_radec.ra, target_radec.dec, s=target_size, c='r')
    # Map
    if center_list is not None:
        if color_list is None: 
            colors = 'lightcoral' if invert else 'limegreen'
        else:
            colors = color_list
        if visibility_list is not None: 
            center_list = center_list[visibility_list]
            if color_list is not None:
                colors = colors[visibility_list]
        add_map(ax, target=center, radec=center_list, 
                        radec_color=colors, target_color='r',
                        legend_color='r' if invert else 'lime',
                        shape_ins=[0.4,0.18])
    return fig, ax

def batch_simulate_image(center_list, step=1, start=0,
                         save_dir='.', image_name='image',
                         target_radec_list=None,
                         epoch_list=None,
                         field_fov=(5.0*u.deg, 5.0*u.deg), 
                         limiting_mag=12.0,
                         target_mag_list=None, invert=False,
                         visibility_list=None, color_list=None):
    """
    Simulates and saves a batch of images based on the provided parameters.

    Parameters:
    center_list (list): List of center coordinates for the images.
    step (int, optional): Step size for iterating through the center_list. Default is 1.
    start (int, optional): Starting index for the iteration. Default is 0.
    save_dir (str, optional): Directory where the images will be saved. Default is '.'.
    image_name (str, optional): Base name for the saved images. Default is 'image'.
    target_radec_list (list, optional): List of target RA/DEC coordinates. Default is None.
    epoch_list (list, optional): List of epochs for the images. Default is None.
    field_fov (tuple, optional): Field of view for the images. Default is (5.0*u.deg, 5.0*u.deg).
    limiting_mag (float, optional): Limiting magnitude for the images. Default is 12.0.
    target_mag_list (list, optional): List of target magnitudes. Default is None.
    invert (bool, optional): Whether to invert the image colors. Default is False.
    visibility_list (list, optional): List of visibility parameters. Default is None.
    color_list (list, optional): List of colors for the targets. Default is None.

    Returns:
    None
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx in tqdm(range(start*step,len(center_list),step)):
        fig, ax = simulate_image(center_list[idx], target_radec=target_radec_list[idx],
                                 field_fov=field_fov, limiting_mag=limiting_mag, 
                                 epoch=epoch_list[idx],
                                 target_mag=target_mag_list[idx],
                                 visibility=visibility_list[idx], invert=invert,
                                 center_list=center_list, visibility_list=visibility_list, 
                                 color_list=color_list)
        fig.savefig(f"{save_dir}/{image_name}-{idx//step}.png", 
                    bbox_inches='tight', pad_inches=0, dpi=200, transparent=False)
        plt.close(fig)
    return None
    

### VIDEO ###

def make_video(image_dir='.', image_name="*.png", save_dir='.', save_name='video', FPS=20):
    """
    Creates a video from a sequence of images.

    Parameters:
    image_dir (str): Directory containing the images. Default is current directory.
    image_name (str): Pattern to match image files. Default is "*.png".
    save_dir (str): Directory to save the output video. Default is current directory.
    save_name (str): Name of the output video file (without extension). Default is 'video'.
    FPS (int): Frames per second for the output video. Default is 20.

    Returns:
    None
    """
    os.makedirs(save_dir, exist_ok=True)
    images = glob.glob(f"{image_dir}/{image_name}")
    images.sort(key=lambda filename: int(''.join(filter(str.isdigit, filename))))
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    video = cv2.VideoWriter(f"{save_dir}/{save_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))  
    for i in tqdm(range(len(images))):
        image = cv2.resize(cv2.imread(images[i]), (width, height))
        video.write(image)  
    video.release()

def compress_video(in_video, out_video, target_size_MB, delete_in=False):
    """
    Compress a video file to a target size.

    Parameters:
    in_video (str): Path to the input video file.
    out_video (str): Path to the output compressed video file.
    target_size_MB (float): Target size of the compressed video in megabytes.
    delete_in (bool): If True, delete the input video file after compression. Default is False.

    Returns:
    None

    Reference:
    https://en.wikipedia.org/wiki/Bit_rate#Encoding_bit_rate
    """
    probe = ffmpeg.probe(in_video)
    # Video duration, in s.
    duration = float(probe['format']['duration'])
    # Target total bitrate, in bps.
    target_total_bitrate = (target_size_MB * 1000 * 1024 * 8) / (1.073741824 * duration)
    # Target video bitrate, in bps.
    video_bitrate = target_total_bitrate
    i = ffmpeg.input(in_video)
    ffmpeg.output(i, out_video,
                  **{'c:v': 'libx264', 'b:v': video_bitrate, 'pass': 1, 'f': 'mp4'}
                  ).overwrite_output().run()
    if delete_in: os.remove(in_video)
    for file in glob.glob("ffmpeg2pass*"): os.remove(file)
