import numpy as np
from numba import jit
import os
import sys

@jit(nopython=True)
def Stats(n, p, i0, j0, L, A):
    """
    Compute a sequence of importance sampling statistics.
    Parameters: n: size of matrix, p: finite field characteristic,
                i0, j0: indexing the pattern group (allowing non-zero entries in (i,j), where i > i0, or i = i0 and j >= j0),
                L: number of iterations, A: starting matrix
    """
    S = np.zeros(L, dtype=np.float64)
    for l in range(L):
        A = Burnside(A, n, p, i0, j0)
        tmp = 0
        if A[i0-1, j0-1] == 0:
            if i0 == n - 1:
                d = DimP(A, n, p, i0, j0)
                tmp = p ** (1 - d)
            else:
                d1 = DimP(A, n, p, i0, j0)
                i, j = NIndex(n, i0, j0)
                d2 = DimP(A, n, p, i, j)
                tmp = p ** (1 - d1 + d2)
        S[l] = tmp
    return S, A

@jit(nopython=True)
def mod_inv(a, p):
    """Compute modular inverse."""
    t, new_t = 0, 1
    r, new_r = p, a
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        return -1
    return t % p

@jit(nopython=True)
def dot_product(a, b):
    """Compute dot product of two arrays."""
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

@jit(nopython=True)
def Gaussian_Elimination_GF(A, p):
    """Perform Gaussian elimination over finite field."""
    m, n = A.shape
    pivot_columns = []
    i, j = 0, 0
    while i < m and j < n:
        sub_matrix = A[i:m, j]
        non_zero_idx = np.nonzero(sub_matrix)[0]
        if len(non_zero_idx) == 0:
            j += 1
            continue
        k = i + non_zero_idx[0]
        entry = A[k, j]
        pivot_columns.append(j)
        A[i, j:], A[k, j:] = A[k, j:].copy(), A[i, j:].copy()
        entry_inv = mod_inv(entry, p)
        A[i, j:] = (A[i, j:] * entry_inv) % p
        for row in range(m):
            if row != i:
                A[row, j:] = (A[row, j:] - A[row, j] * A[i, j:]) % p
        i += 1
        j += 1
    return A, np.array(pivot_columns, dtype=np.int64)

@jit(nopython=True)
def Coordinate(n, i, j, ind):
    """
    Compute the coordinate index in the flattened array based on position (i, j).
    """
    return (2 * n - i) * (i - 1) // 2 + j - i - ind

@jit(nopython=True)
def Burnside(A, n, p, i0, j0):
    """
    One iteration of the Burnside process on the pattern group.
    Parameters: A : input matrix, n: matrix size, p: finite field characteristic,
                i0, j0: indexing the pattern group (allowing non-zero entries in (i,j), where i > i0, or i = i0 and j >= j0)
    """

    ind = (2 * n - i0) * (i0 - 1) // 2 + j0 - i0 - 1
    N = n * (n - 1) // 2 - ind
    M = np.zeros((N,N), dtype=np.int64)

    # Build the matrix M
    for j in range(j0 + 1, n + 1):
        cor_x = Coordinate(n, i0, j, ind)
        for l in range(j0, j):
            cor_y1 = Coordinate(n, l, j, ind)
            cor_y2 = Coordinate(n, i0, l, ind)
            M[cor_x-1, cor_y1-1] = A[i0-1, l-1]%p
            M[cor_x-1, cor_y2-1] = (-A[l-1, j-1])%p

    for i in range(i0 + 1, n):
        for j in range(i + 1, n + 1):
            cor_x = Coordinate(n, i, j, ind)
            for l in range(i + 1, j):
                cor_y1 = Coordinate(n, l, j, ind)
                cor_y2 = Coordinate(n, i, l, ind)
                M[cor_x-1, cor_y1-1] = A[i-1, l-1]%p
                M[cor_x-1, cor_y2-1] = (-A[l-1, j-1])%p

    # Perform Gaussian elimination to obtain Mn and find pivot columns r1
    Mn, r1 = Gaussian_Elimination_GF(M, p)

    r2 = [x for x in np.arange(N) if x not in r1]
    Res = np.zeros(N, dtype=np.int64)

    # Generate random values for r2 indices
    for index in r2:
        Res[index] = np.random.randint(0, p)

    # Back-substitute to solve for the other indices
    for i, index in enumerate(r1):
        Res[index] = (-dot_product(Mn[i, :], Res)) % p

    # Transform Res back into matrix form
    B = np.zeros((n, n), dtype=np.int64)
    for j in range(j0, n+1):
        index = Coordinate(n, i0, j, ind)
        B[i0-1, j-1] = Res[index-1]

    for i in range(i0+1, n+1):
        for j in range(i+1, n+1):
            index = Coordinate(n, i, j, ind)
            B[i-1, j-1] = Res[index-1]

    return B

@jit(nopython=True)
def DimP(A, n, p, i0, j0):
    """
    Dimension of the centralizer.
    Parameters: A : input matrix, n: matrix size, p: finite field characteristic,
                i0, j0: indexing the pattern group (allowing non-zero entries in (i,j), where i > i0, or i = i0 and j >= j0)
    """

    ind = (2 * n - i0) * (i0 - 1) // 2 + j0 - i0 - 1
    N = n * (n - 1) // 2 - ind
    M = np.zeros((N, N), dtype=np.int64)

    for j in range(j0 + 1, n + 1):
        cor_x = Coordinate(n, i0, j, ind)
        for l in range(j0, j):
            cor_y1 = Coordinate(n, l, j, ind)
            cor_y2 = Coordinate(n, i0, l, ind)
            M[cor_x-1, cor_y1-1] = A[i0-1, l-1]
            M[cor_x-1, cor_y2-1] = (-A[l-1, j-1]) % p

    for i in range(i0 + 1, n):
        for j in range(i + 1, n + 1):
            cor_x = Coordinate(n, i, j, ind)
            for l in range(i + 1, j):
                cor_y1 = Coordinate(n, l, j, ind)
                cor_y2 = Coordinate(n, i, l, ind)
                M[cor_x-1, cor_y1-1] = A[i-1, l-1]
                M[cor_x-1, cor_y2-1] = (-A[l-1, j-1]) % p

    M_reduced, _ = Gaussian_Elimination_GF(M, p)
    rank = 0
    for row in range(N):
        if np.any(M_reduced[row, :] != 0):
            rank += 1
    return N-rank

@jit(nopython=True)
def NIndex(n, i0, j0):
    """
    Find the next index in upper triangular matrix
    1 <= i0 < j0 <=n
    """
    if j0 == n:
        return i0 + 1, i0 + 2
    return i0, j0 + 1
