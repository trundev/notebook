import numpy as np
import numpy.typing as npt
import pytest
# Module to be tested
import rlc_funcs
import rlc_reverse


def normalize_angle(angle: npt.NDArray, range=np.pi) -> npt.NDArray:
    """Keep an angle in (-pi, +pi] range"""
    return range - (range - angle) % (2*range)

def normalize_phi(phi: npt.NDArray[np.complexfloating]) -> npt.NDArray:
    """Keep the imaginary component in +/-pi range"""
    return phi.real + 1j*normalize_angle(phi.imag)

@pytest.mark.parametrize('omegas, phis, sample_dt', [
        pytest.param(complex(-.1, np.pi/2), 0, .25, id='Simple attenuating oscillation'),
        *(pytest.param(complex(omega_r, np.pi/2), 0, 1/3, id=f'Attenuation: {omega_r:.3}')
                for omega_r in np.linspace(-5, 5, 10, endpoint=True)),
        *(pytest.param(complex(-.1, omega_i), 0, 1/3, id=f'Frequency: {omega_i:.3}')
                for omega_i in np.linspace(1e-2, 3*np.pi, 10, endpoint=False)), # dt must be less than 1/2 period
    ])
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
    rev_omega, rev_phi = rlc_reverse.from_4samples(sfn, sample_dt, conjugated=conjugated, squeeze=False)
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

@pytest.mark.parametrize('a, b, a0, b0', [
        pytest.param(-.1, np.pi/2, 0, 0, id='Attenuating oscillation'),
        *(pytest.param(-.1, np.pi/2, phi_r, 0, id=f'Attenuation phase {phi_r:.3}')
            for phi_r in np.linspace(-5, 5, 10, endpoint=True)),
        *(pytest.param(-.1, np.pi/2, 0, phi_i, id=f'attenuation aux-phase {phi_i:.3}')
            for phi_i in np.linspace(np.pi, -np.pi, 10, endpoint=False)),   # +/-pi are indistinguishable
    ])
@pytest.mark.parametrize('imag_b', (True, False), ids=('oscillating', 'non-oscillating'))
@pytest.mark.parametrize('add_conj', [True, False], ids=('conjugated', 'simple'))
def test_separated(a, b, a0, b0, imag_b, add_conj, sample_dt=.25):
    if imag_b:
        b *= 1j
        b0 *= 1j
    if add_conj:
        b = b * rlc_reverse.PLUS_MINUS
        b0 = b0 * rlc_reverse.PLUS_MINUS
    test_combined(a + b, a0 + b0, sample_dt, conjugated=add_conj)
