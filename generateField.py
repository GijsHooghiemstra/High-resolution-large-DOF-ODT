# Import all modules
import numpy as np
from scipy.special import spherical_jn, spherical_yn, sph_harm
from pyshtools.expand import spharm_lm


# Define spherical Hankel function of the first kind (not included in SciPy)
def spherical_hn1(n,z,derivative=False):
    return spherical_jn(n,z,derivative)+1j*spherical_yn(n,z,derivative)


def calc_offset_field(name, lambda0, n_med, n_obj, R, Nx, Ny, Lx, Ly, z0, x_offset, numThreads):

    #Physical
    k0 = 2 * np.pi / lambda0 #m^-1
    n_r = n_obj / n_med # [-]

    dx = Lx / Nx #m
    dy = Ly / Ny #m

    x = np.linspace(-Lx/2, Lx/2, Nx) + x_offset
    y = np.linspace(-Ly/2, Ly/2, Ny)

    # Full coordinate matrices
    xx, yy = np.meshgrid(x, y)

        
    # max l=k0*R0; indepent of the units of lambda
    Ltot=int(np.round(k0*R)+0) # single sided L in summation (20 originally)
    # Total number of values for the summation

    # MiePython stopping condition
    mie_size = 2 * np.pi * R / lambda0
    stop_order = int(mie_size + 4.05 * mie_size**0.33333 + 2.0)

    Ltot = stop_order

    l=np.arange(Ltot, dtype=int)

    # Setup arrays
    numerator = np.zeros(Ltot, dtype=complex)
    denominator = np.zeros(Ltot, dtype=complex)
    U = np.zeros((Ny, Nx), dtype=complex)

    for ll in l:
        numerator[ll] = n_r*spherical_jn(ll, k0*R)*spherical_jn(ll, k0*n_r*R, derivative=True) - spherical_jn(ll, k0*R, derivative=True)*spherical_jn(ll, k0*n_r*R)
        denominator[ll] = spherical_jn(ll, k0*n_r*R)*spherical_hn1(ll, k0*R, derivative=True) - n_r*spherical_jn(ll, k0*n_r*R, derivative=True)*spherical_hn1(ll, k0*R)

    # Compute reflection coefficients
    eps = np.finfo(float).eps

    Rl=numerator/(denominator+eps)

    # Unit vector for propagation in spherical coordinates
    theta0 = 0 # Azimuthal propagation direction (undefined for z-direction)
    phi0 = 0 # Polar propagation direction

    r = np.sqrt(np.square(xx) + np.square(yy) + z0*z0)
    theta = np.arctan2(yy, xx)
    phi = np.arctan(np.sqrt(np.square(xx) + np.square(yy)) / z0)

    # Define function to calculate a row of the field
    def calc_order(ll):
        
        U_order = np.zeros((Ny, Nx), dtype=complex)

        for m in range(-ll, ll+1):
            U_order += 4 * np.pi * Rl[ll] * 1j**ll * np.conjugate(spharm_lm(ll, m, phi0, theta0, degrees=False, normalization="ortho")) * spherical_hn1(ll, k0 * r) * spharm_lm(ll, m, phi, theta, degrees=False, normalization="ortho")

        return U_order

    # Multiprocess module used instead of multiprocessing (fixed an issue)
    from multiprocess import Pool

    # Entry point for the program
    if name == '__main__':
        # Create a process pool that uses all cpus
        with Pool(numThreads) as pool:
            # Call the same function with different data in parallel
            for result in pool.imap(calc_order, l):
                # Add the calculated column to the final solution
                U_order = result
                U += U_order

    return U