"""Calculate Euler's function coefficients from its values
"""
import numpy as np

PLUS_MINUS = np.array((1, -1))
EPSILON = 1e-14

def geom_deviation(vals: np.array) -> np.array:
    """Calculate the two "geometric devation" values"""
    geom1 = vals[..., 1:-1]**2 - vals[..., 2:] * vals[..., :-2]
    geom2 = vals[..., 2:-1] * vals[..., 1:-2] - vals[..., 3:] * vals[..., :-3]
    return geom1, geom2

def calc_exp_omega_phi(vals: np.array, conjugated: bool=False) -> np.array:
    """Exponentiated Omega and Phi

    - The last 'vals' axis must contain equidistant samples, each group of 4 adjacents is processed
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

def from_4samples(vals: np.array, dt: float, conjugated: bool=False, squeeze: bool=True):
    """Euler's coefficients from 4 samples at equidistant moments"""
    exp_omega, exp_phi = calc_exp_omega_phi(vals, conjugated)

    # Squeeze last axis, if there is a single result
    if squeeze and exp_omega.shape[-1] == 1:
        exp_omega = np.squeeze(exp_omega, -1)
        exp_phi = np.squeeze(exp_phi, -1)

    # Get exponent coefficients
    omega = np.log(exp_omega) / dt
    phi = np.log(exp_phi)

    # Drop complex result if unnecessary
    # Note that either omega or phi could be real-only, while the other is complex
    if not (omega.imag.any() or phi.imag.any()):
        omega = omega.real
        phi = phi.real
    return omega, phi


#
# Test scenarios
#
if __name__ == '__main__':
    import rlc_funcs

    def normalize_angle(angle: np.array, range=np.pi) -> np.array:
        """Keep an angle in (-pi, +pi] range"""
        return range - (range - angle) % (2*range)

    def normalize_phi(phi: np.array) -> np.array:
        """Keep the imaginary component in +/-pi range"""
        return phi.real + 1j*normalize_angle(phi.imag)

    def test_combined(omegas, phis, sample_dt, num_samples=4, conjugated=False):
        trange = np.arange(num_samples) * sample_dt
        print(f'Omegas {np.round(omegas, 3)}, Phis {np.round(phis, 3)}, Delta t {sample_dt}')
        print(f'  Exp-omegas per sample {np.round(np.exp(omegas * sample_dt), 3)}, exp-phis {np.round(np.exp(phis), 3)}')
        sfn = rlc_funcs.calc_euler_derivs(1, omegas, phis, trange)
        sfn = sfn[0]
        if conjugated:
            sfn = sfn.mean(0)
        else:
            sfn = sfn.real
        print(f'Total sample values {np.round(sfn, 3)}')

        # Invoke reversal function
        rev_omega, rev_phi = from_4samples(sfn, sample_dt, conjugated=conjugated, squeeze=False)
        print(f'Result: Omega {np.round(rev_omega, 3)}, Phi {np.round(rev_phi, 3)}')

        if conjugated:
            assert not np.round(np.exp(rev_phi).mean(0) - sfn[..., :rev_phi.shape[-1]], 12).any(), \
                    f'decomposed Phi mismatch'

        # Rebase Phis to t=0 (was at trange[0])
        rev_phi -= rev_omega * trange[0]
        if rev_omega.ndim:
            # In case of multiple results (num_samples > 4), was at trange[0], trange[1]...
            rev_phi -= (rev_omega * sample_dt) * np.arange(rev_omega.shape[-1])

        # Re-check input values
        rev_sfn = rlc_funcs.calc_euler_derivs(1, rev_omega, rev_phi, trange)
        rev_sfn = rev_sfn[0]
        if conjugated:
            rev_sfn = rev_sfn.mean(0)
        assert not np.round(rev_sfn.real - sfn.real, 8).any(), \
                f'sample value deviation: {np.round(rev_sfn, 3)}, difference {np.round(rev_sfn - sfn, 3)}'

        # Confirm Omega/Phis
        assert not np.round(rev_omega.T - omegas, 8).any(), \
                f'rev_omega deviation: {np.round(rev_omega, 3)}, actual {np.round(omegas, 3)}'
        assert not np.round(normalize_phi(rev_phi.T - phis), 8).any(), \
                f'rev_phi deviation: {np.round(rev_phi, 3)}, actual {np.round(phis, 3)}'
        print(f'---')

    def test_separated(a, b, a0, b0, imag_b, add_conj, sample_dt):
        if imag_b:
            b *= 1j
            b0 *= 1j
        if add_conj:
            b = b * PLUS_MINUS
            b0 = b0 * PLUS_MINUS
        test_combined(a + b, a0 + b0, sample_dt, conjugated=add_conj)

    #
    # Sample "critical" data:
    # two real coefficient sets conjugated and non-conjugated give the same result
    #
    crit_omega1 = np.asarray(-.1 + np.pi/2)
    crit_phi1 = np.asarray(3.8888888888888893 + 2*crit_omega1)
    print(f'Critical data I (from non-conjugated coefficients) {crit_omega1=}, {crit_phi1=}:')
    res1 = rlc_funcs.calc_euler_derivs(1, crit_omega1, crit_phi1, np.arange(4)*.25)[0]
    print(f'  {res1}')

    crit_omega2 = np.log(5 + PLUS_MINUS * 3.872983346207417) / .25
    crit_phi2 = np.log([75.85583107467721, 1775.4170458082385])
    print(f'Critical data II (from conjugated coefficients) {crit_omega2=}, {crit_phi2=}:')
    res2 = rlc_funcs.calc_euler_derivs(1, crit_omega2, crit_phi2, np.arange(4)*.25)[0]
    res2 = res2.mean(0)
    print(f'  {res2}')
    print(f'  {res1-res2=}')

    print('---')
    crit_omega1 = np.asarray(-.1 + np.pi/2)
    crit_phi1 = np.asarray(-5)
    print(f'Critical data I (from non-conjugated coefficients) {crit_omega1=}, {crit_phi1=}:')
    res1 = rlc_funcs.calc_euler_derivs(1, crit_omega1, crit_phi1, np.arange(4)*.25)[0]
    print(f'  {res1}')

    crit_omega2 = np.log(-2 + PLUS_MINUS * 3.4641016151377544 + 0j) / .25
    crit_phi2 = np.log([0.013437587072432266, 3.830692573866725e-05])
    print(f'Critical data II (from conjugated coefficients) {crit_omega2=}, {crit_phi2=}:')
    res2 = rlc_funcs.calc_euler_derivs(1, crit_omega2, crit_phi2, np.arange(4)*.25)[0]
    res2 = res2.mean(0)
    print(f'  {res2.real}')
    print(f'  {res1-res2.real=}')

    print('=== Simple attenuating oscillation ===')
    test_combined(complex(-.1, np.pi/2), 0, .25)
    print('=== Conjugated attenuating oscillation ===')
    test_separated(
            -.1, np.pi/2,
            0, 0,
            imag_b=True, add_conj=True,
            sample_dt=.25)
    print(f'\n=== Range attenuation ===')
    for omega_r in np.linspace(-5, 5, 10, endpoint=True):
        test_combined(complex(omega_r, np.pi/2), 0, 1/3)
    print(f'\n=== Range frequency ===')
    for omega_i in np.linspace(1e-2, 3*np.pi, 10, endpoint=False):  # dt must be less than 1/2 period
        test_combined(complex(-.1, omega_i), 0, sample_dt=1/3)
    # Traverse all imag_b, add_conj combinations: 0-False, 1-True
    for imag_b, add_conj in np.indices((2, 2)).reshape(2,-1).T:
        imag_b, add_conj = bool(imag_b), bool(add_conj)
        title = ', '.join([
                'oscillating' if imag_b else 'non-oscillating',
                'conjugated' if add_conj else 'simple'])
        print(f'\n=== Range attenuation phase:  {title} ===')
        for phi_r in np.linspace(-5, 5, 10, endpoint=True):
            test_separated(
                    -.1, np.pi/2,
                    phi_r, 0,
                    imag_b=imag_b, add_conj=add_conj,
                    sample_dt=.25)
        print(f'\n=== Range attenuation aux-phase: {title} ===')
        for phi_i in np.linspace(np.pi, -np.pi, 10, endpoint=False):    # +/-pi are indistinguishable
            test_separated(
                    -.1, np.pi/2,
                    0, phi_i,
                    imag_b=imag_b, add_conj=add_conj,
                    sample_dt=.25)
