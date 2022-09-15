"""Utility generate number-walls
"""
import itertools
import numpy as np


# Limit the size of a determinants to be calculated at once, in order to:
# - Avoid precision loss, due to huge intermediate values, leading to incorrect result,
#   like: "1e6**3 + 1 - 1e6**3 == 0"
# - Better performance
MAX_DET_SIZE = 3

ASSERT_LEVEL = 1

def calc_row_via_det(data: np.array, degree: int, nan=np.nan) -> np.array:
    """Calculate number-wall row from the first row, by using np.linalg.det()"""
    # Combine number-wall style matrix
    nw_mtx = np.array(tuple(
            data[degree-i: data.shape[-1]-i] for i in range(degree+1)))

    # Get individual determinants (first and last 'index' can NOT be calculated)
    dets = np.full_like(data, nan)
    for i in range(data.shape[-1] - 2*degree):
        dets[i + degree] = np.linalg.det(nw_mtx[..., i:i + degree + 1])
    return dets

def _ref_perm_parity(perm_idxs: np.array) -> np.array:
    """Obtain the parity of permulations from the number of inversions

    Uses nested for-loops, as a reference for pytests
    See:
        https://statlect.com/matrix-algebra/sign-of-a-permutation
        https://en.wikipedia.org/wiki/Parity_of_a_permutation
    """
    odd_mask = False
    # Plain nested for-loops implementation
    for i in range(perm_idxs.shape[-1] - 1):
        for j in range(i + 1, perm_idxs.shape[-1]):
            odd_mask = odd_mask ^ (perm_idxs[..., i] > perm_idxs[..., j])
    return odd_mask

def permutation_parity(perm_idxs: np.array) -> np.array:
    """Obtain the parity of permulations from the number of inversions"""
    # Indices of 'perm_idxs' elements, to calculate inversions/parity at once:
    # the combinations of their number over 2
    idxs = np.fromiter(itertools.combinations(range(perm_idxs.shape[-1]), 2), dtype=(int, 2))

    # Regroup permutation indices in couples to check for inversions
    idxs = perm_idxs[...,idxs]
    odd_mask = np.logical_xor.reduce(idxs[...,0] > idxs[...,1], axis=-1)

    if ASSERT_LEVEL > 3:
        # Use a reference result
        assert (odd_mask == _ref_perm_parity(perm_idxs)).all(), 'Wrong optimized permutation parity'
    return odd_mask

def permutations_indices(size: int) -> np.array:
    """Obtain all permulations for specific number range"""
    # Use int8 as allocation of more than "factorial(128)" elements is impossible anyway
    return np.fromiter(itertools.permutations(range(size)),
            dtype=(np.int8, [size]))

def combinations_parity(comb_mask) -> np.array:
    """Obtain the parity of combinations from the number of inversions"""
    odd_mask = np.logical_xor.accumulate(~comb_mask, axis=-1)
    odd_mask[~comb_mask] = False
    odd_mask = np.logical_xor.reduce(odd_mask, axis=-1)

    if ASSERT_LEVEL > 3:
        # Combine permutation indices for the reference result
        combs = take_by_masks(np.arange(comb_mask.shape[-1]), comb_mask)
        rems = take_by_masks(np.arange(comb_mask.shape[-1]), ~comb_mask)
        perm = np.concatenate((combs, rems), axis=-1)
        # Use a reference result
        assert (odd_mask == _ref_perm_parity(perm)).all(), 'Wrong optimized combinations parity'
    return odd_mask

def combinations_masks(size: int, comb_size: int) -> np.array:
    """Obtain all combinations for specific number range

    Note: The returned mask can be inverted to get the remainder from combination"""
    idxs = np.fromiter(itertools.combinations(range(size), comb_size),
            dtype=(np.int8, [comb_size]))
    masks = np.zeros(shape=(idxs.shape[0], size), dtype=bool)
    masks[np.arange(idxs.shape[0])[...,np.newaxis], idxs] = True
    return masks

def take_by_masks(data: np.array, masks: np.array):
    """Extract elements by masking the last dimension, keeping the mask's shape"""
    # Broadcast the last dimension to match 'masks'
    data = np.broadcast_to(data[...,np.newaxis,:],
            shape=data.shape[:-1] + masks.shape)
    # The reshape() below expects, the masking to extract
    # equal number of elements from each row
    if ASSERT_LEVEL > 1:
        mask_dims = np.count_nonzero(masks, axis=-1)
        assert (mask_dims == mask_dims[0]).all(), 'Mask is not uniform'
    return data[..., masks].reshape(data.shape[:-1] + (-1,))

def det_from_combination(take_data: callable, data_indices: np.array, row_base: int) -> np.array:
    """Matrix determinant calculation on given combinations of indices

        take_data(col_idxs, row_base) - callback to retrieve actual data
            The columns to be read, are grouped in the last "col_idxs" dimension
            The corresponding rows are sequential numbers starting at "row_base"
            Example:
                return matrix[np.arange(col_idxs.shape[-1]) + row_base, col_idxs]
    """
    # Calculate only the determinants, that are small enough
    if data_indices.shape[-1] <= MAX_DET_SIZE:
        # Combine all permutations in a single array
        idxs = permutations_indices(data_indices.shape[-1])
        perms = data_indices[..., idxs]

        res = take_data(perms, row_base)
        del perms
        # Apply determinant rule: products from permutations
        res = res.prod(-1)

        # Get the permutation parity
        odd_masks = permutation_parity(idxs)
        del idxs
    else:
        # Split each determinant into two minor-determinants (from sub-matrices):
        # main (left-side) minor and remainder (right-size) minor
        minor_size = data_indices.shape[-1]//2
        masks = combinations_masks(data_indices.shape[-1], minor_size)
        minors = take_by_masks(data_indices, masks)
        r_minors = take_by_masks(data_indices, ~masks)

        minors = det_from_combination(take_data, minors, row_base)
        r_minors = det_from_combination(take_data, r_minors, row_base + minor_size)
        # Apply determinant rule: products from sub-determinants
        res = minors * r_minors
        del minors, r_minors

        # Get the combination parity
        odd_masks = combinations_parity(masks)
        del masks

    # Apply determinant rule: sum the products by negating the odd-permutations
    # Note:
    # This is preferred, instead of "res[..., odd_masks] *= -1", to prevent from
    # float precision loss in intermediate results during sum()
    even_res = res[..., ~odd_masks]
    odd_res = res[..., odd_masks]
    # First sum the common pairs of even and odd permutations
    comm_size = min(even_res.shape[-1], odd_res.shape[-1])
    res = (even_res[..., :comm_size] - odd_res[..., :comm_size]).sum(-1)
    # Then add the remainder from even or odd ones
    res += even_res[..., comm_size:].sum(-1)
    res -= odd_res[..., comm_size:].sum(-1)
    return res

def calc_row_via_permutations(data: np.array, degree: int, nan=np.nan) -> np.array:
    """Calculate number-wall row from the first row, by using a custom determinant algorithm

    This should be more generic, esp. allow array of Polynomial-s, but for higher 'degree'
    values can be slow/memory consuming.
    """
    def take_data(indices, row_base):
        """Take data from number-wall style matrix"""
        # Convert indices using the number-wall style data-oder
        indices += degree - np.arange(indices.shape[-1]) - row_base
        # Retrieve data for multiple determinants starting at each possible index
        indices = indices + np.arange(data.shape[-1] - 2*degree).reshape((-1,) + (1,) * indices.ndim)
        return data[indices]

    # Split individual determinant permutations into manageable chunks
    res = det_from_combination(take_data, np.arange(degree + 1), 0)
    return np.pad(res, degree, constant_values=nan)

#
# Test scenarios
#
if __name__ == '__main__':
    # Catch over/underflows
    np.seterr(all='raise')

    data = np.arange(20) ** 3.

    num_rows = (data.size + 1) // 2
    print('* Use numpy.linalg.det()')
    for r in range(0, num_rows):
        res = calc_row_via_det(data, r)
        print(f'{r}: {np.round(res, 3)}')
    print()

    print('* Use calculation based on itertools.permutations()')
    nan = np.full_like(data, np.nan, shape=[])
    for r in range(0, num_rows):
        res = calc_row_via_permutations(data, r, nan)
        print(f'{r}: {np.round(res, 3)}')
    print()
