# Jacobi Iteration Method
import numpy as np


def jacobi(A, b, x_o, Tol=1e-3, max_iter=1000):
    n = len(b)
    x = x_o.copy()
    x_new = np.zeros(n)
    iterations_1 = 1

    while iterations_1 < max_iter:
        for i in range(n):
            sum = 0
            for j in range(n):
                if j != i:
                    sum += A[i][j] * x[j]
            x_new[i] = (b[i] - sum) / A[i][i]
        # print(x_new)

        diff = np.linalg.norm(x_new - x, np.inf) / np.linalg.norm(x_new, np.inf)
        if diff < Tol:
            return x_new, iterations_1 + 1

        x = x_new.copy()
        iterations_1 += 1

    return "Maximum # of iterations exceeded", iterations_1


A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]])

b = np.array([6, 25, -11, 15])

x_o = np.array([0, 0, 0, 0])

solution_1, iterations_1 = jacobi(A, b, x_o)
print("Solution 1: ", solution_1)
print("# of iterations:", iterations_1)
print("")


# Gauss Seidel Iteration Method-------------------

def gauss_seidel(A, b, x_o, Tol=1e-3, max_iter=1000):
    x = x_o.copy()
    for k in range(max_iter):
        x_prev = x.copy()
        for i in range(len(A)):
            p = b[i]
            for j in range(len(A)):
                if i != j:
                    p -= A[i][j] * x[j]
            x[i] = p / A[i][i]
        # print(x)

        if np.linalg.norm(x - x_prev, np.inf) / np.linalg.norm(x, np.inf) < Tol:
            break
    return x, k + 1


A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]], dtype=np.float64)

b = np.array([6, 25, -11, 15], dtype=np.float64)
x_o = np.zeros_like(b, dtype=np.float64)

solution_2, iterations_2 = gauss_seidel(A, b, x_o)

print("Solution 2:", solution_2)
print("# of iterations:", iterations_2)
print("")


# Successive Over-relaxation Method---------------------

def SOR(A, b, X_o, w, max_iter, Tol):
    n = len(b)
    x = np.copy(X_o)
    iterations_3 = 0

    while iterations_3 < max_iter:
        x_prev = np.copy(x)
        for i in range(n):
            sum = 0
            for j in range(n):
                if j != i:
                    sum += A[i][j] * x[j]
            x[i] = (1 - w) * x[i] + (w / A[i][i]) * (b[i] - sum)
        # print(x)

        iterations_3 += 1

        if np.linalg.norm(x - x_prev) / np.linalg.norm(x) < Tol:
            return x, iterations_3

    return "Maximum # of iterations exceeded", iterations_3


A = np.array([[4, 3, 0],
              [3, 4, -1],
              [0, -1, 4]])

b = np.array([24, 30, -24])
X_o = np.array([1, 1, 1])
w = 1.25
Tol = 1e-3
max_iter = 1000

solution_3, iterations_3 = SOR(A, b, X_o, w, max_iter, Tol)
print("Solution 3:", solution_3)
# print("# of iterations:", iterations_3)
print("")


##Iterative Refinement Method--------------------

def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        row_index = i
        for j in range(i + 1, n):
            if abs(A[j, i]) > abs(A[row_index, i]):
                row_index = j
        if row_index != i:
            A[[i, row_index]] = A[[row_index, i]]
            b[[i, row_index]] = b[[row_index, i]]
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x


def iter_refine(A, b, N, Tol):
    n = len(b)
    x = gaussian_elimination(A, b)
    k = 1
    cond = np.linalg.norm(x, np.inf)
    while k <= N:
        r = b - np.dot(A, x)
        y = gaussian_elimination(A, r)
        x += y
        if k == 1:
            cond = np.linalg.norm(x, np.inf)
        if np.linalg.norm(y, np.inf) < Tol:
            return x, cond
        k += 1
    return "Maximum iteration exceeded", cond


A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]], dtype=float)

b = np.array([6, 25, -11, 15], dtype=float)
N = 1000
Tol = 1e-3

A = np.array([[4, 3, 0],
              [3, 4, -1],
              [0, -1, 4]], dtype=float)

b = np.array([24, 30, -24], dtype=float)
Tol = 1e-3
N = 1000

solution_4, cond = iter_refine(A, b, N, Tol)
print("Solution 4:", solution_4)
print("Condition number:", cond)