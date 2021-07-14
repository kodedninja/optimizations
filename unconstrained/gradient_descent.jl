include("utils/functions.jl")
include("utils/utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function gradient_descent(f, x0; g=nothing, max_iter=100000, tol=1e-4)
    """
    performs gradient (steepest) descent
    
    the step direction is always the steepest direction, after that the step length is chosen using backtracking
    so that the sufficient decrease condition is fulfilled.

    results in a very slow, but guaranteed convergence.
    """
    xk = x0
    xks = Vector{Vector{Float32}}(); push!(xks, xk)

    f_gxk = g != nothing ? g(x0) : Utils.approx_gradient(f, x0)

    for i in 1:max_iter
        # stopping criteria
        if norm(f_gxk) <= tol
            return xks
        end

        pk = -f_gxk
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 3, 0.9)

        xk = xk + alpha * pk
        push!(xks, xk)

        f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    end

    println("Gradient descent did not converge")
    return xks
end

for i in 1:length(Functions.problems)
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
    xk = gradient_descent(f, s; tol=1e-4)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
