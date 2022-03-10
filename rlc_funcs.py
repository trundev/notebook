"""RLC function calculation

Here the RLC functions are used in the form:
* For regular system (omega_1 != omega_2)
    e^(omega_1*t + phi_1) + e^(omega_2*t + phi_2)
* For critically damped system (omega_1 == omega_2)
    t * e^(omega*t + phi_1) + e^(omega*t + phi_2)
"""
import numpy as np

def solve_quadratic(a: np.array, b: np.array, c: np.array) -> np.array:
    """Calculate the two solutions of quadratic equation"""
    sqrt = np.sqrt(b**2 - 4*a*c)
    return np.stack([(-b + sq) / (2*a) for sq in (sqrt, -sqrt)])

def calc_poly(coefs: np.array, t: np.array) -> complex:
    """Polynomial for the Euler's formula exponent"""
    # Combine arrays in a shape [coefs.shape, t.shape]
    poly_rank = coefs.shape[-1]
    coefs = coefs.reshape(coefs.shape + (1,)*t.ndim)
    t = t ** np.arange(poly_rank).reshape((-1,) + (1,)*t.ndim)
    # Final result in a shape of [coefs.shape[:-1], t.shape]
    return (coefs * t).sum(-t.ndim)

def calc_euler_derivs(num: int, euler_omega: np.array, euler_phi: np.array, t: np.array) -> np.array:
    """Euler's formula from complex frequency and phase"""
    euler_omega = np.asarray(euler_omega)
    t = np.asarray(t)
    exp = np.exp(calc_poly(np.stack(np.broadcast_arrays(euler_phi, euler_omega), axis=-1), t))
    # The result shape will be of [1<num>, omega.shape, t.shape]
    mult = euler_omega ** np.arange(num).reshape((num,) + (1,)*euler_omega.ndim)
    mult = mult.reshape(mult.shape + (1,)*t.ndim)
    return mult * exp

def calc_rlc_fn_derivs(num: int, euler_omega: np.array, euler_phi: np.array, t: np.array) -> np.array:
    """RLC function: sum of Euler's formulas"""
    t = np.asarray(t)
    res = calc_euler_derivs(num, euler_omega, euler_phi, t)
    # Sum along the last dimension (coming from the 'euler_omega'/'euler_phi')
    # Result shape [1<num>, omega.shape[:-1], t.shape]
    return res.sum(-1-t.ndim)

def calc_rlc_fn2_derivs(num: int, euler_omega: np.array, euler_phi: np.array, t: np.array) -> np.array:
    """RLC function for critically damped system: (t+1)*e^(omega*t)"""
    euler_omega = np.asarray(euler_omega)
    t = np.asarray(t)
    res = calc_euler_derivs(num, euler_omega, euler_phi, t)
    #
    # Prepare the array [t + n / omega, 1]
    # the first row will multiplied by omega[...,0], second -- by omega[...,1]
    # finally these will be summed to get ((t + n / omega) * A + B) * e^(omega*t)
    #
    # t component
    t_comp = np.stack((t, np.zeros_like(t)))

    # n/omega component
    n_comp = euler_omega[...,0]
    n_comp = np.arange(num).reshape((num,) + (1,)*n_comp.ndim) / n_comp
    n_comp = np.stack((n_comp, np.ones_like(n_comp)), -1)
    n_comp = n_comp.reshape(n_comp.shape + (-1,)*t.ndim)

    res *= n_comp + t_comp
    return res.sum(-1-t.ndim)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RLC circuit calculator')
    parser.add_argument('--solve-quadratic', nargs=3, metavar=('a', 'b', 'c'),
            help='Solve quadratic equation')
    parser.add_argument('--calc-rlc', nargs=4, metavar=('num', 'omega', 'phi', 't'),
            help='Calculate derivatives of RLC function')
    args = parser.parse_args()

    def args2arr(args):
        """Convert args strings to numpy arrays"""
        return [np.fromstring(v, dtype=complex, sep=',') for v in args]

    if args.solve_quadratic:
        a, b, c = args2arr(args.solve_quadratic)
        print(f'Solutions of {a}*x^2 + {b}*x + {c} = 0:')
        for val in solve_quadratic(a, b, c):
            print(f'  {val}')

    if args.calc_rlc:
        num = int(args.calc_rlc[0])
        omega, phi, t = args2arr(args.calc_rlc[1:])
        # Last dimension holds the two polynomial coefficiens
        if not(omega.size % 2 or phi.size % 2):
            omega = omega[np.newaxis, :]
            phi = phi[np.newaxis, :]
        print(f'Derivatives num={num}, parameters:')
        print(f'  omega={omega}')
        print(f'    (period[sec]={2*np.pi/omega.imag}):')
        print(f'  phi={phi}')
        print(f'  t={t}')
        res = calc_rlc_fn_derivs(num, omega, phi, t)
        print(f'Result (shape={res.shape}):')
        for val in res:
            print(f'  {val}')
