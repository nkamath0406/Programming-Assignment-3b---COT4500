# test_assignment_3.py
import unittest
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

class TestAssignment3(unittest.TestCase):

    def test_gaussian_elimination(self):
        A = np.array([[0.0003, 3.0000], [1.0000, 1.0000]])
        b = np.array([2.0001, 2.0000])
        x = gaussian_elimination(A, b)
        expected = np.array([1.2446380979332121, 0.7553619020667879])
        np.testing.assert_allclose(x, expected, rtol=1e-5)

    def test_lu_factorization(self):
        A = np.array([
            [1, 1, 0, 3],
            [2, 1, -1, 1],
            [3, -1, -1, 2],
            [-1, 2, 3, -1]
        ])
        L, U, det = lu_factorization(A)
        expected_det = 39.0
        self.assertAlmostEqual(det, expected_det, places=4)

        expected_L = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 4.0, 1.0, 0.0],
            [-1.0, -3.0, 0.0, 1.0]
        ])
        np.testing.assert_allclose(L, expected_L, rtol=1e-5)

        expected_U = np.array([
            [1.0, 1.0, 0.0, 3.0],
            [0.0, -1.0, -1.0, -5.0],
            [0.0, 0.0, 3.0, 13.0],
            [0.0, 0.0, 0.0, -13.0]
        ])
        np.testing.assert_allclose(U, expected_U, rtol=1e-5)

    def test_diagonally_dominant(self):
        A = np.array([
            [9, 0, 5, 2, 1],
            [3, 9, 1, 2, 1],
            [0, 1, 7, 2, 3],
            [4, 2, 3, 12, 2],
            [3, 2, 4, 0, 8]
        ])
        self.assertFalse(is_diagonally_dominant(A))

    def test_positive_definite(self):
        A = np.array([
            [2, 2, 1],
            [2, 3, 0],
            [1, 0, 2]
        ])
        self.assertTrue(is_positive_definite(A))

if __name__ == '__main__':
    unittest.main()
