include("utils/functions.jl")
include("utils/utils.jl")

using LinearAlgebra

#import problems and helper functions
import .Functions
import .Utils

function sr1(f, x0; g=nothing, beta=0.5, max_iter=1000, tol=1e-4, r=1e-8)
    """
    sr1 is also a quasi-newton method (as bfgs), but with a different update formula for the inverse hessian.

    the initial approximation is chosen as beta * I and some updates are skipped, which prevents the method from breaking down.

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
    xk = x0
    H = beta * I
	# As asked, the gradient gets approximated and is not given as an argument
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    
    xks = Vector{Vector{Float32}}(); push!(xks, xk)

    for i in 1:max_iter
        # stopping criteria
        if norm(f_gxk) <= tol
            return xks
        end

        pk = -H * f_gxk
		# line search
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
        xnext = xk + alpha * pk
        f_gnext = g != nothing ? g(xnext) : Utils.approx_gradient(f, xnext)
        s = xnext - xk
        y = f_gnext - f_gxk

        d = s - H * y
        if abs((d' * y)) >= r * norm(y) * norm(d)
            H = H + (d * d')/(d' * y)
        end

        xk = xnext
        push!(xks, xk)
        f_gxk = f_gnext
    end
    # if max_iteration is exceeded and the stopping criteria was never met, method did not converge
    println("SR1 did not converge")
    return xks
end

# Go over each problem defined in Functions.problems(functions.jil)
for i in 1:length(Functions.problems)
	# Get obj function, gradient, hessian and starting point
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
	# call method with obj function and starting point (without gradient or hessian, approximated later on)
    xk = sr1(f, s; beta=0.3)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
