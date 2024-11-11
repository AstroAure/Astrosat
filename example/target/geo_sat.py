from pandas import date_range
from astropy.time import Time
import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import satastro

def sat_geo_teme(i=0.0*u.deg, RAAN=90*u.deg):
    epoch = Time(date_range(start="2029-01-01",end="2029-01-04", freq="1min"), format='datetime64', scale='utc')
    a = Earth.R + 35786*u.km
    e = 0.0*u.one
    argp = 0.0*u.deg
    nu = 0*u.deg
    sat_geo = Orbit.from_classical(Earth, a, e, i, RAAN, argp, nu, epoch=epoch[0])
    sat_geo_teme = satastro.orbit_to_TEME(sat_geo, epoch)
    return sat_geo_teme