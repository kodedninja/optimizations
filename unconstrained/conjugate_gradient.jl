include("functions.jl")
include("utils.jl")

using LinearAlgebra

import .Functions
import .Utils

function conjugate_gradient(f, x0; g=nothing, max_iter=1000, tol=1e-4)
    """conjugate gradient by the dai-yuan"""
    multivariate = length(x0) > 1

    xk = x0
    xks = Vector{Vector{Float32}}(); push!(xks, xk)
    fxk = f(xk)
    f_gxk = g != nothing ? g(xk) : Utils.approx_gradient(f, xk)
    pk = -f_gxk

    for i in 1:max_iter
        if norm(f_gxk) <= tol
            return xks
        end

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

    println("CG did not converge")
    return xks
end

for i in 1:length(Functions.problems)
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
    xk = conjugate_gradient(f, s)
    steps = length(xk)
    solution = xk[end]
    println("Steps: $steps\n")
    println("Solution:\n $solution\n")
end
