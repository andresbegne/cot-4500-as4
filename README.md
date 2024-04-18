#Iterative Techniques for Solving Linear Systems

This repository contains python implementations of some methods for matrix algebra. The methods are: 

##1. Jacobi Iterative Method

It is favorable to use this method for diagonally dominant or symmetric positive definite matrices. 

USAGE: solution_1, iterations_1 = jacobi(A, b, x_o)

##2. Gauss-Seidel Method

USAGE: solution_2, iterations_2 = gauss_seidel(A, b, x_o)

##3. Successive Over-relaxation (SOR) Method

USAGE: solution_3, iterations_3 = SOR(A, b, X_o, w, max_iter, Tol)

##4. Iterative Refinement Method

USAGE: solution_4, cond = iter_refine(A, b, N, Tol)

Each method provides the solution to the given linear system along with the number of iterations required for convergence (except for the Iterative Refinement Method, which also provides the condition number).



