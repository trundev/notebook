"""Complex exponent (Euler function) approximator

This is based on the polynomial approximator poly_approx
Includes the experimental logarithmic derivative calculations
"""
import poly_approx
import cmath

DEF_NUM_DERIVS = 4
DEF_EXTRA_DERIVS = 2
DEF_MAX_HISTORY = 8

class deriv_approx(poly_approx.approximator):
    """Approximate complex exponent function by using its real value only"""
    max_rank = None
    num_derivs = None
    derivs = None

    def __init__(self, src=None):
        """Optional-copy constructor"""
        super().__init__(src)
        self.reset_derivs()

    def reset_derivs(self, num_derivs=DEF_NUM_DERIVS, extra_derivs=DEF_EXTRA_DERIVS, max_history=DEF_MAX_HISTORY):
        """Reset derivative history but not poly_approx state"""
        self.num_derivs = num_derivs
        self.max_rank = num_derivs + extra_derivs
        self.max_history = max_history
        self.derivs = {}

    def approximate(self, val, time):
        """Feed approximation data"""
        super().approximate(val, time)
        super().reduce(max_rank=self.max_rank)
        tmp_obj = self.copy()
        d_idx = min(self.num_derivs, tmp_obj.num_deltas()) - 1
        d_time = tmp_obj.get_value_time(d_idx)[1]
        tmp_obj.make_derivs(time=d_time, delta_rank=d_idx)
        # Keep all known derivatives for 'd_time'
        deriv_set = []
        for i in range(tmp_obj.num_deltas()):
            deriv_set.append(tmp_obj.get_value(i, as_deriv=True))
        self.derivs[d_time] = deriv_set
        # Limit the derivative history
        if self.max_history is not None:
            while len(self.derivs) > self.max_history:
                key_iter = iter(self.derivs.keys())
                del self.derivs[next(key_iter)]

    def iter_derivs(self, min_rank=0):
        """Iterator over the calculated derivatives"""
        for t, deriv_set in self.derivs.items():
            if len(deriv_set) >= min_rank:
                yield t, deriv_set

    def reversed_iter_derivs(self, min_rank=0):
        """Iterator over the calculated derivatives"""
        for t, deriv_set in reversed(self.derivs.items()):
            if len(deriv_set) >= min_rank:
                yield t, deriv_set

#
# Experimental
#
def calc_omegas_from_4derivs(derivs):
    """Experimental calculation of Omega from rlc_circuit.ipynb"""
    # Deduce real component of exponent
    a = derivs[3]*derivs[0] - derivs[2]*derivs[1]
    a /= 2 * (derivs[2]*derivs[0] - derivs[1]*derivs[1])
    # Deduce imaginary component of exponent
    b = cmath.sqrt(
            derivs[3]**2 * derivs[0]**2
            -6 * derivs[3] * derivs[2] * derivs[1] * derivs[0]
            -3 * derivs[2]**2 * derivs[1]**2
            +4 * derivs[3] * derivs[1]**3
            +4 * derivs[2]**3 * derivs[0]
        )
    #NOTE: Use abs() to ensure the two results are in descending order (+/-)
    b /= 2 * abs(derivs[2]*derivs[0] - derivs[1]*derivs[1])
    return [a + b, a - b]

def calc_3log_derivs(derivs):
    """Experimental calculation of the first 3 logarithmic derivatives

    See: https://www.mathcha.io/editor/MvEN9FnktejfjvTEle5O5fyylgoYSW3e3ljingB3YP
    [Logarithmic derivative from regular ones]
    """
    # f^L = f' / f
    first = derivs[1]/derivs[0]
    # f^{LL} = f'' / f' - f^L
    second = derivs[2]/derivs[1] - first

    #
    # Formulas for the third derivative
    #
    # f^{LLL} = \frac{f''' / f' - 3 f'' / f + 2 (f^L)^2}{ f^{LL} } - f^{LL}
    third = (derivs[3]/derivs[1] - 3*derivs[2]/derivs[0] + 2*first**2) / second - second

    # Return the function value as index 0
    return derivs[0], first, second, third

def calc_omegas_from_log_derivs(log_derivs):
    """Experimental calculation of Omega by using logarithmic derivatives

    See: https://www.mathcha.io/editor/MvEN9FnktejfjvTEle5O5fyylgoYSW3e3ljingB3YP
    [Logarithmic derivative implications]
    """
    # a = f^L + \frac{f^{LLL} + f^{LL}}{2}
    a = log_derivs[1] + (log_derivs[3] + log_derivs[2]) / 2
    # b = \sqrt{\frac{( f^{LLL} + f^{LL} )^2}{4} + f^{LL}f^L}
    b = cmath.sqrt(
            (log_derivs[3] + log_derivs[2]) ** 2 / 4
            + log_derivs[2] * log_derivs[1] )
    return [a + b, a - b]
