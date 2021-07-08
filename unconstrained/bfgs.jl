include("functions.jl")
include("utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function bfgs(f, x0; g=nothing, max_iter=1000, tol=1e-4)
    """
    bfgs

    bfgs is a quasti-newton method. it tries to calculate and maintain an approximation of the true hessian (or inverse hessian)
    without knowing this.

    in this implementation, the initial inverse hessian approximation is calculated according to the first step,
    and the updates are skipped if the curvature condition is not fulfilled.
    """
    xk = x0
    H = I
    xks = Vector{Vector{Float32}}(); push!(xks, xk)       
    
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)

    for i in 1:max_iter
        if norm(f_gxk) <= tol
            return xks
        end

        pk = -H * f_gxk
        alpha = Utils.backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
        xnext = xk + alpha * pk
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
    
    println("BFGS did not converge")
    return xks
end


for i in 1:length(Functions.problems)
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
    xk = bfgs(f, s)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
