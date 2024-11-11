# Satastro - Astronomy for and with satellites

[![PyPI - Version](https://img.shields.io/pypi/v/satastro)](https://pypi.org/project/satastro/)

## Description

Satastro is a Python package to combine astronomy and satellites. It enables to calculate the RA-Dec position of a satellite or asteroid in the sky when viewed from Earth. It also enables to calculate these astronomical positions when the observer is a satellite orbiting the Earth.

Satastro also provides a simple to use interface with [NASA JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) to retrieve ephemerides of Small Solar System Bodies.

![Apophis asteroid ground trace](https://github.com/AstroAure/Satastro/blob/main/figures/apopis_ground-trace.png?raw=true)

With Satastro, one can easily plot visibility maps for asteroids or satellites, along with night times. It is also offers the possibility to plot star field with the UCAC4 or GaiaDR3 catalogs to find the target in the sky.

![Sky plot of Apophis viewed from a SSO satellite](https://github.com/AstroAure/Satastro/blob/main/figures/sky-plot.png?raw=true)

## Origin

This package has been developed for the mission analysis of a Cubesat mission to follow the Apophis asteroid flyby of the Earth in April 2029. The objective was to simulate what the satellite would see when the asteroid would fly close to it, and size the attitude control and payload of the satellite.

![Apophis asteroid RA-Dec position in the sky viewed from SSO](https://github.com/AstroAure/Satastro/blob/main/figures/apopis_from_sat.png?raw=true)

## Installation

This package can be very easily installed using pip :

 ```pip install satastro```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See the LICENSE file for details.