include("functions.jl")
include("utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function newton(f, x0; g=nothing, h=nothing, max_iter=50, tol=1e-4)
    """
    performs newton's method

    next to the gradient, the inverse hessian is also used to define the step direction.
    the step length is again chosen by backtracking, starting from 1.

    because of the second-order information of the hessian, convergence is blazing fast close to the solution.
    """
    multivariate = length(x0) > 1

    xk = x0
    xks = Vector{Vector{Float32}}(); push!(xks, xk)
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, x0)

    for i in 1:max_iter
        if norm(f_gxk) <= tol
            return xks
        end

        f_hxk = h != nothing ? h(xk) : Utils.approx_hessian(f, xk)
        
        if multivariate
            pk = -inv(f_hxk) * f_gxk
        else
            pk = [-1/f_hxk * f_gxk]
        end
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
        
        xk = xk + alpha * pk
        push!(xks, xk)
        f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    end
    
    println("Newton's method did not converge")
    return xks
end

for i in 1:length(Functions.problems)
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
    xk = newton(f, s)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
