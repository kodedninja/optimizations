include("utils/functions.jl")
include("utils/utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function conjugate_gradient(f, x0; g=nothing, max_iter=1000, tol=1e-4)
    """conjugate gradient by the dai-yuan

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
    fxk = f(xk)
	# As asked, the gradient gets approximated and is not given as an argument
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    pk = -f_gxk

    for i in 1:max_iter
        # stopping criteria
        if norm(f_gxk) <= tol
            return xks
        end
		# line search
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)

        xnext = xk + alpha * pk
        f_gnext = g != nothing ? g(xnext) : Utils.approx_gradient(f, xnext)
        
        beta = (norm(f_gnext)^2)/((f_gnext - f_gxk)' * pk)
        pnext = -f_gnext + beta * pk

        xk = xnext
        push!(xks, xk)
        f_gxk = f_gnext
        pk = pnext
    end
	# if max_iteration is exceeded and the stopping criteria was never met, method did not converge
    println("CG did not converge")
    return xks
end

# Go over each problem defined in Functions.problems (utils/functions.jl)
for i in 1:length(Functions.problems)
	# Get obj function, gradient, hessian and starting point
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
	# call method with obj function and starting point (without gradient or hessian, approximated later)
    xk = conjugate_gradient(f, s)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
