
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    M = np.hstack((A.astype(float), b.reshape(-1, 1)))

    for i in range(n):
        for j in range(i+1, n):
            if M[i][i] == 0:
                continue
            ratio = M[j][i] / M[i][i]
            M[j, i:] = M[j, i:] - ratio * M[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    return x

def lu_factorization(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    U = A.astype(float).copy()

    for i in range(n):
        L[i, i] = 1
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] = U[j, i:] - factor * U[i, i:]

    det = np.prod(np.diag(U))
    return L, U, det

def is_diagonally_dominant(A):
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True

def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def main():
    # Q1 - Gaussian Elimination
    A1 = np.array([[0.0003, 3.0000], [1.0000, 1.0000]])
    b1 = np.array([2.0001, 2.0000])
    x1 = gaussian_elimination(A1, b1)
    print("Q1 Solution:", x1)

    # Q2 - LU Factorization
    A2 = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ])
    L, U, det = lu_factorization(A2)
    print("Q2 Determinant:", det)
    print("Q2 L Matrix:\n", L)
    print("Q2 U Matrix:\n", U)

    # Q3 - Diagonally Dominant
    A3 = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    print("Q3 Diagonally Dominant:", is_diagonally_dominant(A3))

    # Q4 - Positive Definite
    A4 = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    print("Q4 Positive Definite:", is_positive_definite(A4))

if __name__ == "__main__":
    main()
