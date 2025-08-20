import numpy as np
from numba import njit
from numba.typed import Dict
from numba import types

@njit
def numba_unique_cell(anno, mask):
    # Initialize a typed dictionary for storing unique values and counts
    unique_counts = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    
    # Parallel iteration over 3D array
    for i in range(anno.shape[0]):
        for j in range(anno.shape[1]):
                val = anno[i, j]
                if val < 1:
                    continue
                
                if val in unique_counts:
                    unique_counts[val][0] += 1
                else:
                    unique_counts[val] = np.array([1, 0], dtype=np.int64)
                
                if mask[i, j] > 0:
                    unique_counts[val][1] += 1
    
    return unique_counts

@njit
def numba_unique_vessel(anno, mask, skel):
    # Initialize a typed dictionary for storing unique values and counts
    unique_counts = Dict.empty(key_type=types.int64, value_type=types.int64[:])
    
    # Parallel iteration over 3D array
    for i in range(anno.shape[0]):
        for j in range(anno.shape[1]):
            val = anno[i, j]
            if val < 1:
                continue
            
            if val in unique_counts:
                unique_counts[val][0] += 1
            else:
                unique_counts[val] = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
            
            if mask[i, j] > 0:
                unique_counts[val][1] += 1
                
            if skel[i, j] > 0:
                unique_counts[val][2] += 1
            
            if skel[i, j] > 1:
                unique_counts[val][3] += 1

    return unique_counts
