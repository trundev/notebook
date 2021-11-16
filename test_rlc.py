"""Test by using RLC sircuit simulation"""
import cmath
import extrapolator

NUM_DERIVS = 4

def euler_formula(z: complex, t: float) -> complex:
    """Calculates e^(a+ib)t, a=1/tau, b=omega, t=time"""
    return cmath.e ** (z * t)

def euler_derivative(z: complex, t: float, level: int = 0) -> complex:
    """Calculates derivative of euler_formula()"""
    return z ** level * cmath.e ** (z * t)

def euler_full_formula(z: complex, z0: complex, t: float) -> complex:
    """Calculates e^{(a+ib)t+(c+id)}, a=1/tau, b=omega, t=time"""
    return cmath.e ** ((z * t) + z0)

def euler_full_derivative(z: complex, z0: complex, t: float, level: int = 0) -> complex:
    """Calculates derivative of euler_full_formula()"""
    return z ** level * cmath.e ** ((z * t) + z0)

# First derivative of ratio:
#   Partial[Divide[Partial[Power[e,ax]cos\(40)bx\(41),x],Power[e,ax]cos\(40)bx\(41)],x]
#   https://www.wolframalpha.com/input/?i=Partial%5BDivide%5BPartial%5BPower%5Be%2Cax%5Dcos%5C%2840%29bx%5C%2841%29%2Cx%5D%2CPower%5Be%2Cax%5Dcos%5C%2840%29bx%5C%2841%29%5D%2Cx%5D
# Second derivative of ratio:
#   D[Divide[Partial[Power[e,ax]cos\(40)bx\(41),x],Power[e,ax]cos\(40)bx\(41)],{x,2}]
# Ratio:
# Divide[D[Divide[Partial[Power[e,ax]cos\(40)bx\(41),x],Power[e,ax]cos\(40)bx\(41)],{x,2}],Partial[Divide[Partial[Power[e,ax]cos\(40)bx\(41),x],Power[e,ax]cos\(40)bx\(41)],x]]


COMPLEX_FMT = '{0.real:8.3f},{0.imag:8.3f}'
fmt_complex = lambda x: COMPLEX_FMT.format(x)

def compare_result(v, epsilon=1e-14):
    if isinstance(v, complex):
        ok = max(v.real, v.imag) <= epsilon and min(v.real, v.imag) >= epsilon
    else:
        ok = abs(v) <= epsilon
    return 'ok' if ok else 'FAIL: %s'%(v)

# Number of derivatives, required by calc_euler_imag()
CALC_EULER_IMAG_DERIVS = 4

def calc_euler_imag(derivs):
    """Calculate imaginary component of Euler's function from its derivatives"""
    # Numerator part of 'logd_deriv_logd'
    imag_val = (
            derivs[3] * derivs[0] ** 2
            -3 * derivs[2] * derivs[1] * derivs[0]
            +2 * derivs[1] ** 3
        )
    # Numerator part of 'b'
    imag_val /= cmath.sqrt(
            -derivs[3] ** 2 * derivs[0] ** 2
            +6 * derivs[3] * derivs[2] * derivs[1] * derivs[0]
            +3 * derivs[2] ** 2 * derivs[1] ** 2
            -4 * derivs[3] * derivs[1] ** 3
            -4 * derivs[2] ** 3 * derivs[0]
        ).real
    ###HACK: Negate imaginary component, like -sqrt() instead of +sqrt()
    imag_val = -imag_val
    return imag_val

def find_ab_test(euler_omega, euler_phi = None):
    """Test the formula to find the coefficients of Euler's function"""
    print('Euler omega:', fmt_complex(euler_omega),
            '' if euler_phi is None else ', Euler phi: %s'%fmt_complex(euler_phi) )
    print(20*'-')

    approx_in = extrapolator.approximator()
    approx_clog = extrapolator.approximator()

    for t in range(24):
        t /= 6
        if euler_phi is None:
            func = euler_formula(euler_omega, t)
            derivs = [euler_derivative(euler_omega, t, i) for i in range(NUM_DERIVS)]
        else:
            func = euler_full_formula(euler_omega, euler_phi, t)
            derivs = [euler_full_derivative(euler_omega, euler_phi, t, i) for i in range(NUM_DERIVS)]

        print('{0:.3f}:{1} - {2} check {3}'.format(t,
                fmt_complex(func), fmt_complex(derivs[0]), compare_result(func-derivs[0], 0)))

        print('  ratio {0}'.format(fmt_complex(derivs[1] / func)))
        r_ratio = derivs[1].real / func.real

        # The angle for the transcendent functions, used as a reference
        ref_angle = euler_omega.imag * t
        if euler_phi is not None:
            ref_angle += euler_phi.imag

        # Deriv/function ratio: $a-b*tan(bx)$
        our_val = euler_omega.real - euler_omega.imag * cmath.tan(ref_angle).real
        print('  real-ratio {0:.3f} - {1:.3f} check {2}'.format(r_ratio, our_val, compare_result(r_ratio - our_val)))

#        # First derivative: $a*e^{ax}cos(bx)-b*e^{ax}sin(bx)$
#        our_val = euler_omega.real * (cmath.e ** (euler_omega.real * t) * cmath.cos(ref_angle)).real \
#                - euler_omega.imag * (cmath.e ** (euler_omega.real * t) * cmath.sin(ref_angle)).real
#        print('  real-first {0:.3f} - {1:.3f} check {2}'.format(derivs[1].real, our_val, compare_result(derivs[1].real - our_val)))
#
#        # Second derivative: $(a^2-b^2)e^{ax}cos(bx) - 2ab*e^{ax}sin(bx)$
#        our_val = (euler_omega.real ** 2 - euler_omega.imag ** 2) * (cmath.e ** (euler_omega.real * t) * cmath.cos(ref_angle)).real \
#                - 2 * euler_omega.real * euler_omega.imag * (cmath.e ** (euler_omega.real * t) * cmath.sin(ref_angle)).real
#        print('  real-second {0:.3f} - {1:.3f} check {2}'.format(derivs[2].real, our_val, compare_result(derivs[2].real - our_val)))
#
#        # Derivative of the "ratio": $\frac{f''(x)}{f(x)} - \frac{f'^2(x)}{f^2(x)}$
#        our_val = derivs[2].real / derivs[0].real - derivs[1].real ** 2 / derivs[0].real ** 2
#        # Reference val: $-\frac{b^2}{cos^2(bx)}$
#        ref_val = - euler_omega.imag ** 2 / (cmath.cos(ref_angle) ** 2).real
#        print('    real-first of ratio {0:.3f} - {1:.3f} check {2}'.format(ref_val, our_val, compare_result(ref_val - our_val)))
#
#        # Second derivative of the "ratio": $\frac{f'''(x)}{f(x)} - 3\frac{f''(x)f'(x)}{f^2(x)} + 2\frac{f'^3(x)}{f^3(x)}$
#        our_val = derivs[3].real / derivs[0].real \
#                - 3 * derivs[2].real * derivs[1].real / derivs[0].real ** 2 \
#                + 2 * derivs[1].real ** 3 / derivs[0].real ** 3
#        # Reference val: $-2\frac{b^3sin(bx)}{cos^3(bx)}$
#        ref_val = -2 * euler_omega.imag ** 3 * (cmath.sin(ref_angle) / cmath.cos(ref_angle) ** 3).real
#        print('    real-second of ratio {0:.3f} - {1:.3f} check {2}'.format(ref_val, our_val, compare_result(ref_val - our_val)))

#        # Ratio between second and first above
#        our_val = (
#                derivs[3].real / derivs[0].real
#                - 3 * derivs[2].real * derivs[1].real / derivs[0].real ** 2
#                + 2 * derivs[1].real ** 3 / derivs[0].real ** 3
#            )
#        our_val /= derivs[2].real / derivs[0].real - derivs[1].real ** 2 / derivs[0].real ** 2
#        # Reference val:
#        ref_val = 2 * euler_omega.imag * cmath.tan(ref_angle).real
#        print('      ratio {0:.3f} - {1:.3f} check {2}'.format(ref_val, our_val, compare_result(ref_val - our_val)))
#        #assert abs(our_val - ref_val) < 1e-10, f'{our_val - ref_val=}'
#
#        must_be_a = derivs[1].real/derivs[0].real + our_val/2
#        print(f'        {must_be_a=:.3f}', compare_result(must_be_a - euler_omega.real))
#        print()
#
#        # Simplify the expression (to avoid division by zero)
#        logd_deriv_logd_val = (
#                derivs[3].real * derivs[0].real ** 2
#                -3 * derivs[2].real * derivs[1].real * derivs[0].real
#                +2 * derivs[1].real ** 3
#            )
#        logd_deriv_logd_val /= derivs[0].real * (derivs[2].real * derivs[0].real - derivs[1].real ** 2)
#        print('      logd_deriv_logd_val {0:.3f} - {1:.3f}'.format(logd_deriv_logd_val, ref_val), compare_result(logd_deriv_logd_val - ref_val))

        #
        # Find a - Real component of Euler's omega (inverted time-constant)
        #
        must_be_a = (
                derivs[3].real * derivs[0].real
                - derivs[2].real * derivs[1].real
            )
        must_be_a /= 2 * (derivs[2].real * derivs[0].real - derivs[1].real ** 2)
        print(f'        {must_be_a=:.3f}', compare_result(must_be_a - euler_omega.real))
        assert -1e-14 < must_be_a - euler_omega.real < 1e-14
        print()

        #
        # Find b - Imaginary component of Euler's omega (angular speed)
        #
        must_be_b = (
                -derivs[3].real ** 2 * derivs[0].real ** 2
                +6 * derivs[3].real * derivs[2].real * derivs[1].real * derivs[0].real
                +3 * derivs[2].real ** 2 * derivs[1].real ** 2
                -4 * derivs[3].real * derivs[1].real ** 3
                -4 * derivs[2].real ** 3 * derivs[0].real
            )
        must_be_b = cmath.sqrt(must_be_b).real
        must_be_b /= 2 * abs(derivs[2].real * derivs[0].real - derivs[1].real ** 2)
        print(f'        {must_be_b=:.3f}', compare_result(must_be_b - euler_omega.imag))
        assert -1e-14 < must_be_b - euler_omega.imag < 1e-14
        print()

        #
        # Find the imaginary component of Euler's function
        #
        must_be_imag = calc_euler_imag([f.real for f in derivs[:4]])
        print(f'        {must_be_imag=:.3f}', compare_result(must_be_imag - func.imag))
        assert -1e-13 < must_be_imag -  func.imag < 1e-13
        print()

        #
        # Test the derivatives calculated by extrapolator.approximate()
        #
        approx_in.approximate(func.real, t)
        print('Approximated derivs:')
        min_dif = cmath.inf, 0
        for _, tmp_t in approx_in:
            print(f'  at t={tmp_t}:')
            tmp_obj = approx_in.copy()
            tmp_obj.make_derivs(tmp_t)
            approx_derivs = []
            for rank in range(tmp_obj.num_deltas()):
                approx = tmp_obj.get_value(rank, as_deriv=True)
                actual = euler_full_derivative(euler_omega, euler_phi, tmp_t, rank).real
                print(f'    {approx:8.3f}, actual {actual:8.3f}')
                approx_derivs.append(approx)
            if len(approx_derivs) >= CALC_EULER_IMAG_DERIVS:
                approx_imag = calc_euler_imag(approx_derivs)
                actual = euler_full_formula(euler_omega, euler_phi, tmp_t).imag
                print(f'  * imag val: {approx_imag:8.3f}, actual {actual:8.3f}', compare_result(approx_imag - actual))
                diff = abs((approx_imag - actual) / actual)
                if min_dif[0] > diff:
                    min_dif = diff, tmp_t
        print(f'* done, diff {min_dif[0]*100:.1f}% at t={min_dif[1]} (behind {t - min_dif[1]})')
        print('='*10)

    return 0

def test_rlc():
    find_ab_test(complex(-cmath.pi / 6, cmath.pi), complex(2, -1))

if __name__ == '__main__':
    if test_rlc():
        exit(1)
