include("functions.jl")
include("optimizations.jl")

import .Functions
import .Optimizations

function solve(f, g, h, s)
    """solves problem f with all methods from starting point s"""
    min_gd = Optimizations.gradient_descent(f, g, s)
    min_newton = Optimizations.newton(f, g, h, s)
    min_cg = Optimizations.conjugate_gradient(f, g, s)
    min_quasi_newton = Optimizations.sr1(f, g, s)

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


f, g, h, s = Functions.problems[9]
min_gd = Optimizations.gradient_descent(f, g, s)
println(length(min_gd))
min_gd = Optimizations.gradient_descent(f, nothing, s)
println(length(min_gd))
exit()

for i in 1:length(Functions.problems)
    f, g, h, s = Functions.problems[i]

    println("Problem $i\nStarting point: $s\n")
    min_gd, min_newton, min_cg, min_qn = solve(f, g, h, s)
    steps_gd, steps_newton, steps_cg, steps_qn = length(min_gd), length(min_newton), length(min_cg), length(min_qn)
    min_gd, min_newton, min_cg, min_qn = min_gd[end], min_newton[end], min_cg[end], min_qn[end]
    println("Steps:\n Steepest descent: $steps_gd \n Newton: $steps_newton \n Conjugate gradient: $steps_cg \n Quasi-Newton: $steps_qn\n")
    println("Solutions:\n SD: $min_gd\n Newton: $min_newton\n CG: $min_cg\n QN: $min_qn")

    println()
end
