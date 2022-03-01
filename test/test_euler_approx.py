import sys
import cmath
#import pytest
import pandas as pd

 # Module to be tested
import euler_approx


if True:
    # Oscillating sample
    TEST_DATA = 'falstad-data/100mH_10_15uF-capacitor.txt.csv'
    TEST_CAPACITANCE = 15e-6    # 15uF
    TEST_RESISTANCE = 10        # 10Ohm
    TEST_INDUCTANCE = .1        # 100mH
elif True:
    # Non-oscillating sample
    TEST_DATA = 'falstad-data/100mH_200_15uF-capacitor.txt.csv'
    TEST_CAPACITANCE = 15e-6    # 15uF
    TEST_RESISTANCE = 200       # 200Ohm
    TEST_INDUCTANCE = .1        # 100mH
#else: TODO: Critically damped sample

def is_equal(val: complex, ref_val: complex, epsilon: float) -> bool:
    """Compare float/complex value with epsilon"""
    return abs(val - ref_val) < epsilon

def test_simple():
    """Simple approximation check"""
    df = pd.read_csv(TEST_DATA, index_col='time')

    # Intentionally reduce the sample rate (each 100-th)
    df = df[pd.Index(range(df.size)) % 100 == 0]

    margin = 1e-12
    min_derivs = 4
    obj = euler_approx.deriv_approx()
    for time, volt in df.itertuples():
        print(f'{time:.5f}: {volt}')
        obj.approximate(volt, time)
        for idx, (t, deriv_set) in enumerate(obj.iter_derivs()):
            print(f'  {idx}:', t, deriv_set)
            assert is_equal(deriv_set[0], df.loc[t][0], margin), \
                    f'Zero-deriative {deriv_set[0]} does not match actual data {df.loc[t][0]}'
        print('* reversed')
        for idx, (t, deriv_set) in enumerate(obj.reversed_iter_derivs(min_derivs)):
            print(f'  {idx}:', t, deriv_set)
            assert len(deriv_set) >= 4, f'Returned less derivatives {len(deriv_set)} than requested {min_derivs}'
        assert t <= time, f'Derivative time {t} is after the data {time}'

def test_derivatives():
    """Calculate voltages on resistor and inductor and check the sum"""
    df = pd.read_csv(TEST_DATA, index_col='time')

    # Derivative approximator with history limit
    obj = euler_approx.deriv_approx()
    obj.reset_derivs(num_derivs=3, max_history=2)

    margin = .05
    for t, volt in df.itertuples():
        print(f'{t:.5f}: {volt}')
        obj.approximate(volt,t)
        #
        # Derive voltages on resistor and inductor
        #
        for t, deriv_set in obj.reversed_iter_derivs(3):
            r_volt = deriv_set[1] * TEST_CAPACITANCE * TEST_RESISTANCE
            l_volt = deriv_set[2] * TEST_CAPACITANCE * TEST_INDUCTANCE
            volt_sum = deriv_set[0] + r_volt + l_volt
            print(f'  Voltage sum at {t:.5f}: {deriv_set[0]:.1f}{r_volt:+.1f}{l_volt:+.1f} = {volt_sum:.2f}')
            assert is_equal(volt_sum, 0, margin), \
                f'Nonzero voltage sum at {t}: {deriv_set[0]:.2f}{r_volt:+.2f}{l_volt:+.2f} = {volt_sum:.3f}'

def solve_quadratic(a:complex, b:complex, c:complex) -> complex:
    """Calculate the two solutions of quadratic equation"""
    sqrt = cmath.sqrt(b**2 - 4*a*c)
    return [(-b + sq) / (2*a) for sq in (sqrt, -sqrt)]

def omega_repr(omega:complex) -> str:
    """String representation of complex omega (Euler's exponent)"""
    return f'{omega=} -> tau {1/omega.real} sec, frequency {omega.imag/(2*cmath.pi)} Hz'

def test_calc_omega():
    """Check the Euler's omega calculation"""
    df = pd.read_csv(TEST_DATA, index_col='time')

    ref_omegas = solve_quadratic(TEST_INDUCTANCE, TEST_RESISTANCE, 1/TEST_CAPACITANCE)
    print('Expected:', '\n\t'.join(omega_repr(omega) for omega in ref_omegas))

    # Intentionally reduce the sample rate (each 10-th)
    df = df[pd.Index(range(df.size)) % 10 == 0]

    # Derivative approximator with history limit
    obj = euler_approx.deriv_approx()
    obj.reset_derivs(num_derivs=4, max_history=1)

    margin = .5
    for t, volt in df.itertuples():
        print(f'{t:.5f}: {volt}')
        obj.approximate(volt,t)
        #
        # Derive Euler's omega
        #
        # need extra derivative to have a value closer to the expected
        for t, deriv_set in obj.reversed_iter_derivs(4 + 2):
            omegas = euler_approx.calc_omegas_from_4derivs(deriv_set)
            print(f'  {t:.5f}:', '\n\t'.join(omega_repr(omega) for omega in omegas))
            for omega, ref_omega in zip(omegas, ref_omegas):
                assert is_equal(omega, ref_omega, margin), \
                        f'Calculated omega {omega} does not match the expectation {ref_omega}'

def test_log_derivatives():
    """Calculate logarithmic derivatives, validate by using omega"""
    df = pd.read_csv(TEST_DATA, index_col='time')

    ref_omegas = solve_quadratic(TEST_INDUCTANCE, TEST_RESISTANCE, 1/TEST_CAPACITANCE)
    print('Expected:', '\n\t'.join(omega_repr(omega) for omega in ref_omegas))

    # Intentionally reduce the sample rate (each 10-th)
    df = df[pd.Index(range(df.size)) % 10 == 0]

    # Derivative approximator with history limit
    obj = euler_approx.deriv_approx()
    obj.reset_derivs(num_derivs=4, max_history=1)

    margin = .5
    for t, volt in df.itertuples():
        print(f'{t:.5f}: {volt}')
        obj.approximate(volt,t)

        # need extra derivatives to have a value closer to the expected
        for t, deriv_set in obj.reversed_iter_derivs(4 + 2):
            #
            # Get the logarithmic derivatives
            #
            log_derivs = euler_approx.calc_3log_derivs(deriv_set)
            print(f'  {t:.5f}:', log_derivs)

            omegas = euler_approx.calc_omegas_from_log_derivs(log_derivs)
            print(f'  {t:.5f}:', '\n\t'.join(omega_repr(omega) for omega in omegas))
            for omega, ref_omega in zip(omegas, ref_omegas):
                assert is_equal(omega, ref_omega, margin), \
                        f'Calculated omega {omega} does not match the expectation {ref_omega}'

#
# For non-pytest debugging
#
if __name__ == '__main__':
    res = test_simple()
    if res:
        sys.exit(res)
    res = test_derivatives()
    if res:
        sys.exit(res)
    res = test_calc_omega()
    if res:
        sys.exit(res)
    res = test_log_derivatives()
    if res:
        sys.exit(res)
