import sys
import numpy as np
import pytest

 # Module to be tested
import number_wall
number_wall.ASSERT_LEVEL = 5


def test_basic():
    """Basic tests"""
    data = np.arange(8) ** 2.

    # Base row
    ref_res = data
    res = number_wall.calc_row_via_det(data, 0)
    np.testing.assert_allclose(res, ref_res, err_msg='Zero-row must be unchaged')

    res = number_wall.calc_row_via_permutations(data, 0, -1)
    np.testing.assert_equal(res, ref_res, err_msg='Zero-row must be unchaged')

    # First row
    ref_res = data[1:-1] ** 2 - data[:-2] * data[2:]
    ref_res = np.pad(ref_res, 1, constant_values=np.nan)
    res = number_wall.calc_row_via_det(data, 1)
    np.testing.assert_allclose(res, ref_res, equal_nan=True, err_msg='Wrong first row')

    res = number_wall.calc_row_via_permutations(data, 1)
    np.testing.assert_allclose(res, ref_res, equal_nan=True, err_msg='Wrong first row')

def test_asserts():
    """Provoked failures"""
    number_wall.ASSERT_LEVEL = 10

    # take_by_masks() expects uniform mask
    masks = np.array([[False, False], [True, False]])
    with pytest.raises(AssertionError):
        number_wall.take_by_masks(np.zeros_like(masks), masks)

    # Complete determinant calculations cause float overflows
    def sample_dets():
        degree = 4
        res = number_wall.calc_row_via_permutations(np.arange(16, dtype=float)**degree, degree+1)
        return res[degree+1:-degree-1]
    # Some regular data test
    res = sample_dets()
    np.testing.assert_allclose(res, 0, err_msg='Non-zero determinant')
    # Same data calculated w/o sub-determinants
    keep_ds = number_wall.MAX_DET_SIZE
    number_wall.MAX_DET_SIZE = 10
    res = sample_dets()
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(res, 0, err_msg='Expected float overflow')
    number_wall.MAX_DET_SIZE = keep_ds

def test_main():
    """Main tests with higher degree matrices"""
    data = np.arange(20) ** 4. / 64

    # Use calc_row_via_det() as a reference
    nan = np.full_like(data, np.nan, shape=[])
    for degree in range(6):
        print(f'Testing matrix degree {degree}')
        ref_res = number_wall.calc_row_via_det(data, degree)

        res = number_wall.calc_row_via_permutations(data, degree, nan)
        print(f'  {res}')

        np.testing.assert_allclose(res, ref_res, equal_nan=True, atol=1e-6,
                err_msg='Wrong determinant {degree}')

def test_complex():
    """Tests with complex(oscillating) exponent functions"""
    # Select a random seed, but print it to allow exact reproduction
    seed = np.random.randint(1<<16)
    print(f'* Random seed {seed}')
    rng = np.random.default_rng(seed)

    # Three random oscillations with decay
    exps = rng.random(3) * 2j * np.pi
    exps -= rng.random(exps.shape)
    print(f'Inividual rotations [deg]: {np.round(exps.imag * 180 / np.pi, 1)}')
    data = np.exp(exps * np.arange(20)[:, np.newaxis]).mean(-1)

    for degree in range(exps.size - 1, exps.size + 2):
        print(f'Testing {exps.size} exponents with matrix degree {degree}')
        res = number_wall.calc_row_via_permutations(data, degree)
        # Cut the leading/trailing NaN-s
        res = res[degree:-degree]
        if degree == exps.size - 1:
            # When matrix size is equal to the number of exponents,
            # determinants must form a geometric progression
            rotation = res[1:] / res[:-1]
            print(f'  Result rotation steps [deg]: {np.round(np.angle(rotation, deg=True), 1)}')
            np.testing.assert_allclose(rotation, rotation[0])
        elif degree >= exps.size:
            # when matrix size is larger than the number of exponents,
            # determinants must be zero
            np.testing.assert_allclose(res, 0, atol=1e-16)

#
# For non-pytest debugging
#
if __name__ == '__main__':
    res = test_basic()
    if res:
        sys.exit(res)
    #res = test_asserts()
    #if res:
    #    sys.exit(res)
    res = test_main()
    if res:
        sys.exit(res)
    res = test_complex()
    if res:
        sys.exit(res)
