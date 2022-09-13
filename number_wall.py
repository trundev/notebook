"""Utility generate number-walls
"""
import itertools
import numpy as np

def calc_row_via_det(data: np.array, index: int, nan=np.nan) -> np.array:
    """Calculate number-wall row from the first row, by using np.linalg.det()"""
    arr = np.full_like(data, nan, shape=(index+1,) + data.shape)
    # Populate data along array-rows
    for off in range(arr.shape[0]):
        arr[index - off, ..., :data.shape[-1] - off] = data[..., off:]
    # Get individual determinants (first and last 'index' can NOT be calculated)
    dets = np.full_like(data, nan)
    for i in range(data.shape[-1] - 2*index):
        dets[i + index] = np.linalg.det(arr[..., i:i+arr.shape[0]])
    return dets

def permutation_parity(perm: np.array) -> np.array:
    """Obtain the parity of permulations from the number of inversions

    See:
        https://statlect.com/matrix-algebra/sign-of-a-permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation
    """
    odd_mask = False
    for i in range(perm.shape[-1] - 1):
       for j in range(i + 1, perm.shape[-1]):
            odd_mask = odd_mask ^ (perm[..., i] > perm[..., j])
    return odd_mask

def calc_row_via_permutations(data: np.array, index: int, nan=np.nan) -> np.array:
    """Calculate number-wall row from the first row, by using itertools.permutations()

    This should be more generic, esp. allow array of Polynomial-s, but for higher 'index'
    values can be slow/memory consuming.
    """
    #
    # Approach I:
    # Convert permutation for numpy.array - memory consuming:
    #   index == 11 allocates ~5GB intermediate array
    #
    # Array of all permutations, shape is: (factorial(index+1), index+1)
    # (use int8 as allocation of more than "factorial(128)" elements is impossible anyway)
    perm = np.fromiter(itertools.permutations(range(index + 1)), dtype=(np.int8, [index + 1]))
    odd_mask = permutation_parity(perm)
    # Convert to number-wall style indices, pointing to the original data
    perm += index - perm[0]
    # Extend permutations to calculate all determinants at once,
    # shape is: (result_size, factorial(index+1), index+1)
    perm = perm + np.arange(data.shape[-1] - 2*index)[..., np.newaxis, np.newaxis]
    res = data[perm].prod(axis=-1)
    # Sum the products (polynomials) by negating the odd-permutations
    res[..., odd_mask] *= -1
    res = res.sum(-1)
    # The result is padded with nan-s (np.full_like converts 'np.nan' to 'int' if necessary)
    return np.pad(res, index, constant_values=nan)

#
# Test scenarios
#
if __name__ == '__main__':
    # Catch over/underflows
    np.seterr(all='raise')

    data = np.arange(20) ** 3.

    print('* Use numpy.linalg.det()')
    for r in range(0, 5):
        res = calc_row_via_det(data, r)
        print(f'{r}: {np.round(res, 3)}')
    print()

    print('* Use calculation based on itertools.permutations()')
    nan = np.full_like(data, np.nan, shape=[])
    for r in range(0, 5):
        res = calc_row_via_permutations(data, r, nan)
        print(f'{r}: {np.round(res, 3)}')
    print()
