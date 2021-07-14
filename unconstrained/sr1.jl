include("utils/functions.jl")
include("utils/utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function sr1(f, x0; g=nothing, beta=0.5, max_iter=1000, tol=1e-4, r=1e-8)
    """
    sr1
    
    sr1 is also a quasi-newton method (as bfgs), but with a different update formula for the inverse hessian.

    the initial approximation is chosen as beta * I and some updates are skipped, which prevents the method from breaking down.
    """
    xk = x0
    H = beta * I
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    
    xks = Vector{Vector{Float32}}(); push!(xks, xk)

    for i in 1:max_iter
        # stopping criteria
        if norm(f_gxk) <= tol
            return xks
        end

        pk = -H * f_gxk

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
    
    println("SR1 did not converge")
    return xks
end

for i in 1:length(Functions.problems)
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
    xk = sr1(f, s; beta=0.3)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
