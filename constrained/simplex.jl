using LinearAlgebra

function simplex_step!(A, b, c, x, basic, nonbasic)
    B = A[:, basic]
    N = A[:, nonbasic]

    c_b = c[basic]
    c_n = c[nonbasic]

    lambda = inv(B') * c_b
    s_n = c_n - N' * lambda

    if all(s_n .>= 0)
        return true
    end

    q = argmin(s_n) # entering index
    d = inv(B) * A[:, q]

    println("q=", q)
    println("d=", d)

    if all(d .<= 0)
        error("simplex: unbounded problem")
    end

    ratios = x[basic][d .> 0] ./ d[d .> 0]
    p = argmin(ratios) # leaving index
    x_q = ratios[p]

    # update x
    x[basic] .= x[basic] - d * x_q
    x[nonbasic] .= 0
    x[q] = x_q

    # update the basic set
    basic[p] = q

    return false
end

function simplex(A, b, c; max_steps=100)
    m, n = size(A)

    basic = [3, 4]
    nonbasic = collect(setdiff(BitSet(1:n), BitSet(basic)))

    B = A[:, basic]
    N = A[:, nonbasic]
    B_inv = inv(B)

    x = zeros(n)
    x[basic] .= B_inv * b
    x[nonbasic] .= 0

    solution = false
   
    for i in 1:max_steps
        if simplex_step!(A, b, c, x, basic, nonbasic)
            println("value ", c' * x)
            return x
        end
        println(x, basic)
        println("value ", c'*x)
        exit()

        nonbasic = collect(setdiff(BitSet(1:n), BitSet(basic)))
    end
    
    error("simplex: could not find solution in max steps")
end

A = [1 1 1 0; 2 0.5 0 1]
b = [5, 8]
c = [-4, -2, 0, 0]

simplex(A, b, c)
