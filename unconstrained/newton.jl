include("utils/functions.jl")
include("utils/utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function newton(f, x0; g=nothing, h=nothing, max_iter=50, tol=1e-4)
    """
    performs newton's method

    next to the gradient, the inverse hessian (through LU factorization) is also used to define the step direction.
    the step length is again chosen by backtracking, starting from 1.

    because of the second-order information of the hessian, convergence is blazing fast close to the solution.

    f:
    Objective function

    x0:
    Starting point

    g:
    Gradient of f, can either be given as an argument when calling the function or
    estimated via Utils.approx_gradient() if g=nothing

    max_iteration:
    Maximum number of iteration the algortihm is allowed to perform

    tol:
    Tolerance for the stopping criteria
    """
    multivariate = length(x0) > 1

    xk = x0
    xks = Vector{Vector{Float32}}(); push!(xks, xk)
    # As asked, the gradient gets approximated and is not given as an argument
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, x0)

    for i in 1:max_iter
        # stopping criteria
        if norm(f_gxk) <= tol
            return xks
        end
        # As asked, the hessian gets approximated and is not given as an argument
        f_hxk = h != nothing ? h(xk) : Utils.approx_hessian(f, xk)
        
        if multivariate
			# uses LU decomposition to get the inverse matrix
            pk = -(lu(f_hxk) \ f_gxk)
        else
            pk = [-1/f_hxk * f_gxk]
        end
		# line search
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
        
        xk = xk + alpha * pk
        push!(xks, xk)
        f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    end

    # if max_iteration is exceeded and the stopping criteria was never met, method did not converge
    println("Newton's method did not converge")
    return xks
end

# Go over each problem defined in Functions.problems (utils/functions.l)
for i in 1:length(Functions.problems)
	# Get obj function, gradient, hessian and starting point
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
	# call method with obj function and starting point (without gradient or hessian, approximated later)
    xk = newton(f, s)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
