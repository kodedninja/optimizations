include("functions.jl")
include("optimizations.jl")

import .Functions
import .Optimizations

function solve(f, g, h, s)
    """solves problem f with all methods from starting point s"""
    min_gd = Optimizations.gradient_descent(f, g, s)
    min_newton = Optimizations.newton(f, g, h, s)
    min_cg = Optimizations.conjugate_gradient(f, g, s)
    min_quasi_newton = Optimizations.bfgs(f, g, s)

    return min_gd, min_newton, min_cg, min_quasi_newton
end

function generate_multivariate(n)
    """generates and solves a random problem with n variables and prints the underlying matrix, solution, starting point tuple"""

    A, r = Functions.random_multivariate_pair(n)

    f, g, h = Functions.problem_from_multivariate_pair(A, r)
    s = r + randn(n) .* 10

    min_gd, min_newton, min_cg, min_qn = solve(f, g, h, s)

    println("Problem:")
    println(A, "\n\n", r, "\n\n", r + s, "\n")
    println("GD")
    println(length(min_gd), " steps")
    println(min_gd[end])
    println("Newton")
    println(length(min_newton), " steps")
    println(min_newton[end])
    println("CG")
    println(length(min_cg), " steps")
    println(min_cg[end])
    println("Quasi-Newton")
    println(length(min_qn), " steps")
    println(min_qn[end])
end

f, g, h = Functions.problem_from_multivariate_pair(Functions.A4, Functions.r4)
s = Functions.s4

min_gd, min_newton, min_cg, min_qn = solve(f, g, h, s)

println("Problem:")
println("GD")
println(length(min_gd), " steps")
println(min_gd[end])
println("Newton")
println(length(min_newton), " steps")
println(min_newton[end])
println("CG")
println(length(min_cg), " steps")
println(min_cg[end])
println("Quasi-Newton")
println(length(min_qn), " steps")
println(min_qn[end])
