import math
import numpy as np

## === CONSTANTS ===

# Speed of light
C             = 3.e10                         # cm/s

# Planck's h constant
H_PLANCK      = np.float64( 4.136e-18 )       # keV s

# Electron radius
R_e           = 2.83e-13                      # cm

# Mass of proton
M_p           = np.float64( 1.673e-24 )       # g

##----------------------------------------------------------
## Constants for converting things

MICRON2CM     = 1.e-6 * 100.                  # cm <-> micron
PC2CM         = 3.09e18                       # cm <-> pc
ANGS2MICRON   = 1.e-10 * 1.e6                 # A <-> micron

ARCS2RAD      = (2.0*np.pi) / (360.*60.*60.)  # rad <-> arcsec
ARCM2RAD      = (2.0*np.pi) / (360.*60.)      # rad <-> arcmin

HC            = (C * H_PLANCK)                # keV cm
HC_ANGS       = (C * H_PLANCK) * 1.e8         # keV angs

##----------------------------------------------------------
## Cosmology related constants

# Hubble's constant
H0            = 75.                           # km/s/Mpc

# Critical density for Universe
RH0_CRIT      = np.float64(1.1e-29)

# Density in units of RH0_CRIT
OMEGA_D       = 1.e-5                         # dust
OMEGA_M       = 0.3                           # matter
OMEGA_L       = 0.7                           # dark energy

# c/H term in distance integral (a distance)
# c/H = Mpc, then convert to cm
CperH0        = (C * 1.e-5 / H0) * (1.e6 * PC2CM)



h      = 6.62607e-34    # 6.62607004 x 10-34 m2 kg / s (or J s^-1)
c      = 2.998e8        # m s^-1
kB     = 1.38065e-23    # 1.38064852 x 10-23 m2 kg s-2 K-1
kappa0 = 0.192          # kg m^-2
nu0    = 856.5e9        # Hz
Msun   = 1.989e30       # kg

# Hubble's constant
# H0    = 73
H0     = 75.                           # km/s/Mpc