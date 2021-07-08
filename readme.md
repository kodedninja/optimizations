# optimizations

A few optimization methods implemented for those (like me) who understand and learn through code. Homework for a Numerical Optimizations course, following the numerical optimization bible by Nocedal & Wright.

The `unconstrained` contains the unconstrained minimization methods gradient descent, Newton's method, conjugate gradient, and the quasi-Newton methods, BFGS and SR1. Each file contains the algorithm implementation and some code to run it on some problems. `unconstrained/utils` contains `functions.jl`, a set of problems and helpers to generate these, and `utils.jl`, utilities shared across the methods: backtracking line search, gradient & Hessian approximation.
