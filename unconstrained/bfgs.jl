include("utils/functions.jl")
include("utils/utils.jl")

using LinearAlgebra

#import problems and helper functions
import .Functions
import .Utils

function bfgs(f, x0; g=nothing, max_iter=1000, tol=1e-4)
    """
    bfgs is a quasti-newton method. it tries to calculate and maintain an approximation of the true hessian (or inverse hessian)
    without knowing this.

    in this implementation, the initial inverse hessian approximation is calculated according to the first step,
    and the updates are skipped if the curvature condition is not fulfilled.

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
    H = I
    xks = Vector{Vector{Float32}}(); push!(xks, xk)       
    
	# As asked, the gradient gets approximated and is not given as an argument
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)

	# Algorithm 6.1 from the book 
    for i in 1:max_iter
        # stopping criteria
        if norm(f_gxk) <= tol
            return xks
        end

        pk = -H * f_gxk
		# line search
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
        xnext = xk + alpha * pk
		# As asked, the hessian gets approximated and is not given as an argument
        f_gnext = g != nothing ? g(xnext) : Utils.approx_gradient(f, xnext)
        s = xnext - xk
        y = f_gnext - f_gxk

        if i == 1
            H = (y' * s)/(y' * y) * I
        end
					
        if y'*s > 0
            rho = 1/(y' * s)
            H = (I - rho * s * y') * H * (I - rho * y * s') + rho * s * s'
        end

        xk = xnext
        push!(xks, xk)
        f_gxk = f_gnext
    end
    
	# if max_iteration is exceeded and the stopping criteria was never met, method did not converge
    println("BFGS did not converge")
    return xks
end

# Go over each problem defined in Functions.problems(functions.jil)
for i in 1:length(Functions.problems)
	# Get obj function, gradient, hessian and starting point
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
	# call method with obj function and starting point (without gradient or hessian, approximated later on)
    xk = bfgs(f, s)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
