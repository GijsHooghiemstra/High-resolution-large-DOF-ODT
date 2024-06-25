# Calculates the scattered field for two concentric cylinders
# with n_obj_c for center cylinder and n_obj_s for the outer shell cylinder.
# Based upon the works of A. J. Devaney, Mathematical foundations of imaging
# tomography and wavefield inversion, Chapter 6.3.4


# Import all modules
import numpy as np
from scipy.special import jv, hankel1, jvp, h1vp
import matplotlib.pyplot as plt

def besselder(l, ka, type):
    # make sure to import from scipy.special import jv and hankel1 first
    if type==0:     # Bessel
        besselder=jv(l-1,ka) - (l/ka)*jv(l,ka)
        # or equivalently
        #besselder=(l/ka)*jv(l,ka)-jv(l+1,ka)
        return besselder
    else:   # Hankel
        besselder=hankel1(l-1,ka) - (l/ka)*hankel1(l,ka)
        # or equivalently
        # besselder=(l/ka)*hankel1(l,ka)-hankel1(l+1,ka)
        return besselder

def calc_field(name, lambda0, n_med, n_obj, R, Nx, Ny, Lx, Ly, z0):

    #Physical
    k0 = 2 * np.pi / lambda0 #m^-1

    nr = n_obj# / n_med # [-]

    dx = Lx / Nx #m
    dy = Ly / Ny #m

    x = np.linspace(-Lx/2, Lx/2, Nx)
    y = np.linspace(-Ly/2, Ly/2, Ny)

    # Full coordinate matrices
    xx, yy = np.meshgrid(x, y)

        
    # MiePython stopping condition
    mie_size = 2 * np.pi * R / lambda0
    stop_order = int(mie_size + 4.05 * mie_size**0.33333 + 2.0)
    Ltot=2*stop_order+1
    l=np.linspace(stop_order-Ltot+1, Ltot-stop_order-1, Ltot, dtype=int)
    # Total number of values for the summation

    # Setup arrays
    U = np.zeros((Ny, Nx), dtype=complex)

    # Unit vector for propagation in cylindrical coordinates
    phi0 = 0 # Polar propagation direction

    # Detector space coordinates
    r = np.sqrt(np.square(xx) + np.square(z0))
    phi = np.arctan(xx/z0)

    # Setup arrays
    numerator = np.zeros(Ltot, dtype=complex)
    denominator = np.zeros(Ltot, dtype=complex)

    for ll in l:
        numerator[ll] = n_r*jv(ll, k0*R)*jv(ll, k0*n_r*R, derivative=True) - jv(ll, k0*R, derivative=True)*jv(ll, k0*n_r*R)
        denominator[ll] = jv(ll, k0*n_r*R)*hankel1(ll, k0*R, derivative=True) - n_r*jv(ll, k0*n_r*R, derivative=True)*hankel1(ll, k0*R)

    # Compute reflection coefficients
    eps = np.finfo(float).eps
    Rl=numerator/(denominator+eps)

    for ll in l:
        U += 1j**ll * np.exp(-1j*ll * phi0) * Rl[ll] * hankel1(ll, k0*r) * np.exp(1j*ll*phi)

    return U