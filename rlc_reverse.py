"""Calculate Euler's function coefficients from its values
"""
import numpy as np
import numpy.typing as npt


PLUS_MINUS = np.array((1, -1))
EPSILON = 1e-14

def geom_deviation(vals: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Calculate the two "geometric deviation" values"""
    geom1 = vals[..., 1:-1] * vals[..., 1:-1] - vals[..., 2:] * vals[..., :-2]
    geom2 = vals[..., 2:-1] * vals[..., 1:-2] - vals[..., 3:] * vals[..., :-3]
    return geom1, geom2

def calc_exp_omega_phi(vals: npt.NDArray, conjugated: bool=False
                       ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Exponentiated Omega and Phi

    - The last 'vals' axis must contain equidistant samples, each group of 4 adjacent is processed
    - To obtain actual Omega and Phi, take logarithms, then divide Omega's by delta-t
    - Each Phi assumes its position is t=0. To rebase all to time-zero, use:
        exp_phi /= exp_omega ** (np.arange(exp_omega.shape[-1]) + time0/delta_t)
    """
    assert vals.shape[-1] >= 4, 'Need at-least 4 values'
    geom1, geom2 = geom_deviation(vals)

    old_err = np.seterr(divide='ignore', invalid='ignore')

    exp_omega_x = geom2 / (2 * geom1[..., :-1])
    exp_omega_y = np.sqrt(exp_omega_x**2 - geom1[..., 1:] / geom1[..., :-1], dtype=complex)
    # Ensure the sqrt(-<n>) is positive imaginary, to keep the order of conjugates
    # Note that, for some negative values, with NZERO imaginary part: "sqrt(-<n>-0j).imag < 0"
    exp_omega_y[(exp_omega_y.real == 0) & (exp_omega_y.imag < 0)] *= -1

    if conjugated:
        # Place +/- in axis 0
        exp_omega_y = (PLUS_MINUS * exp_omega_y.T[..., np.newaxis]).T

    exp_phi = vals[..., :-3] - (exp_omega_x * vals[..., :-3] - vals[..., 1:-2]) / exp_omega_y

    # Identify and patch the non-conjugated real-only results
    # Note that in this case 'geom1'-s are zero, but this has more precision
    scal = vals[..., 1:] / vals[..., :-1]
    mask = np.abs(scal[..., 1:] - scal[..., :-1]) < EPSILON
    mask = mask[..., 1:] & mask[..., :-1]
    if mask.any():
        # Alternatively: np.mean((scal[:-2], scal[1:-1], scal[2:]), 0)
        exp_omega_x[mask] = scal[..., 1:-1][mask]
        exp_omega_y[..., mask] = 0
        exp_phi[..., mask] = vals[..., :-3][mask]

    np.seterr(**old_err)
    return exp_omega_x + exp_omega_y, exp_phi

def from_4samples(vals: npt.NDArray, dt: float, conjugated: bool=False, squeeze: bool=True
                  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Euler's coefficients from 4 samples at equidistant moments"""
    exp_omega, exp_phi = calc_exp_omega_phi(vals, conjugated)

    # Squeeze last axis, if there is a single result
    if squeeze and exp_omega.shape[-1] == 1:
        exp_omega = np.squeeze(exp_omega, -1)
        exp_phi = np.squeeze(exp_phi, -1)

    # Get exponent coefficients
    omega = np.log(exp_omega) / dt
    phi: npt.NDArray[np.floating] = np.log(exp_phi)

    # Drop complex result if unnecessary
    # Note that either omega or phi could be real-only, while the other is complex
    if not (omega.imag.any() or phi.imag.any()):
        omega = omega.real
        phi = phi.real
    return omega, phi
