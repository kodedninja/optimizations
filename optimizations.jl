"""
implementations of optimization methods

all functions stop when the norm of the gradient at point xk is closer to 0 than the tolerance parameter (1e-4).
"""


module Optimizations
    using LinearAlgebra

    import ..Functions
  
    function backtracking_line_search(f, f_gxk, xk, pk, alpha_init, rho; c=1e-4)
        """
        Armijo backtracking

        walks back from an initial step length alpha_init until the sufficient decrease condition is fulfilled
        """
        alpha = alpha_init
        fxk = f(xk)

        while f(xk + alpha * pk) > fxk + c*alpha * f_gxk' * pk
            alpha = rho * alpha
        end

        return alpha
    end

    function gradient_descent(f, g, x0; max_iter=100000, tol=1e-4)
        """
        performs gradient (steepest) descent
        
        the step direction is always the steepest direction, after that the step length is chosen using backtracking
        so that the sufficient decrease condition is fulfilled.

        results in a very slow, but guaranteed convergence.
        """
        xk = x0
        xks = Vector{Vector{Float32}}(); push!(xks, xk)

        f_gxk = g != nothing ? g(x0) : Functions.approx_gradient(f, x0)

        for i in 1:max_iter
            if norm(f_gxk) <= tol
                return xks
            end

            pk = -f_gxk
            alpha = backtracking_line_search(f, f_gxk, xk, pk, 3, 0.9)
            
            xk = xk + alpha * pk
            push!(xks, xk)

            f_gxk = g != nothing ? g(xk) : Functions.approx_gradient(f, xk)
        end
        
        println("Gradient descent did not converge")
        return xks
    end

    function newton(f, g, h, x0; max_iter=50, tol=1e-4)
        """
        performs newton's method

        next to the gradient, the inverse hessian is also used to define the step direction.
        the step length is again chosen by backtracking, starting from 1.

        because of the second-order information of the hessian, convergence is blazing fast close to the solution.
        """
        multivariate = length(x0) > 1

        xk = x0
        xks = Vector{Vector{Float32}}(); push!(xks, xk)
        f_gxk = g(xk)

        for i in 1:max_iter
            if norm(f_gxk) <= tol
                return xks
            end
            
            if multivariate
                pk = -inv(h(xk)) * f_gxk
            else
                pk = [-1/h(xk) * f_gxk]
            end
            alpha = backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
            
            xk = xk + alpha * pk
            push!(xks, xk)
            f_gxk = g(xk)
        end
        
        println("Newton's method did not converge")
        return xks
    end

    function conjugate_gradient(f, g, x0; max_iter=1000, tol=1e-4)
        """conjugate gradient by the dai-yuan"""
        multivariate = length(x0) > 1

        xk = x0
        xks = Vector{Vector{Float32}}(); push!(xks, xk)
        fxk = f(xk)
        f_gxk = g(xk)
        pk = -f_gxk

        for i in 1:max_iter
            if norm(f_gxk) <= tol
                return xks
            end

            alpha = backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)

            xnext = xk + alpha * pk
            f_gnext = g(xnext)
            
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

    function bfgs(f, g, x0; max_iter=1000, tol=1e-4)
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
        
        f_gxk = g(xk)

        for i in 1:max_iter
            if norm(f_gxk) <= tol
                return xks
            end

            pk = -H * f_gxk
            alpha = backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)

            xnext = xk + alpha * pk
            f_gnext = g(xnext)
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

    function sr1(f, g, x0; beta=0.5, max_iter=1000, tol=1e-4, r=1e-8)
        """
        sr1
        
        sr1 is also a quasi-newton method (as bfgs), but with a different update formula for the inverse hessian.

        the initial approximation is chosen as beta * I and some updates are skipped, which prevents the method from breaking down.
        """
        xk = x0
        H = beta * I
        f_gxk = g(xk)

        xks = Vector{Vector{Float32}}(); push!(xks, xk)

        for i in 1:max_iter
            if norm(f_gxk) <= tol
                return xks
            end

            pk = -H * f_gxk

            alpha = backtracking_line_search(f, f_gxk, xk, pk, 1, 0.9)
            xnext = xk + alpha * pk
						
            f_gnext = g(xnext)

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
end
