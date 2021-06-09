"""
implementations of optimization methods
"""

module Optimizations
    using LinearAlgebra
  
    function backtracking_line_search(f, f_gxk, xk, pk, alpha_init, rho, c=1e-4)
        """Armijo backtracking"""
        alpha = alpha_init
        fxk = f(xk)

        while f(xk + alpha * pk) > fxk + c*alpha * f_gxk' * pk
            alpha = rho * alpha
        end

        return alpha
    end

    function zoom(phi, phi_g, alpha_low, alpha_high, c1, c2, interp=0.5, max_iter=100)
        """zoom function for wolfe line search"""
        phi_zero = phi(0)
        phi_gzero = phi_g(0)

        for i in 1:max_iter
            alpha = interp * (alpha_low + alpha_high)
            y = phi(alpha)

            if (y > phi_zero + c1 * alpha * phi_gzero) || y >= phi(alpha_low)
                alpha_high = alpha
            else
                dy = phi_g(alpha)

                if abs(dy) <= -c2 * phi_gzero
                    return alpha
                end
                if dy * (alpha_high - alpha_low) >= 0
                    alpha_high = alpha_low
                end
                alpha_low = alpha
            end
        end

        return (alpha_high + alpha_low) / 2
    end

    function wolfe_line_search(phi, phi_g, alpha_zero, alpha_max, c1=1e-4, c2=0.9, interp=0.5)
        """
        linear line search for getting the optimal step size imposing the wolfe conditions

        phi(alpha) = f(xk + alpha * pk)
        """

        alpha_last = alpha_zero
        alpha = interp * (0. + alpha_max)
        
        phi_zero = phi(0.)
        phi_gzero = phi_g(0.)
        y = phi(alpha)
        y_last = phi(alpha_last)        

        i = 1
        while true
            y = phi(alpha)
            if y > phi_zero + c1 * alpha * phi_gzero || (i > 1 && y >= phi(alpha_last))
                return zoom(phi, phi_g, alpha_last, alpha, c1, c2)
            end

            dy = phi_g(alpha)

            if abs(dy) <= -c2 * phi_gzero
                return alpha
            end

            if dy >= 0
                return zoom(phi, phi_g, alpha, alpha_max, c1, c2)
            end

            # prepare next loop
            alpha_last = alpha
            alpha = interp * (alpha + alpha_max)
            y_last = y
            y = phi(alpha)
            i += 1
        end
    end

    function gradient_descent(f, g, x0, max_iter=100000, tol=1e-4)
        """performs gradient descent with backtracking line search"""
        xk = x0
        xks = Vector{Vector{Float32}}(); push!(xks, xk)
        f_gxk = g(x0)

        for i in 1:max_iter
            if norm(f_gxk) <= tol
                return xks
            end

            pk = -f_gxk
            alpha = backtracking_line_search(f, f_gxk, xk, pk, 3, 0.9)
            
            xk = xk + alpha * pk
            push!(xks, xk)
            f_gxk = g(xk)
        end
        
        println("gradient descent did not converge")
        return xks
    end

    function newton(f, g, h, x0, max_iter=50, tol=1e-4)
        """performs newton's method with backtracking line search"""
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
        
        println("newton's method did not converge")
        return xks
    end

    function conjugate_gradient(f, g, x0, max_iter=1000, tol=1e-4)
        """conjugate gradient by the dai-yuan method using backtracking line search"""
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

    function bfgs(f, g, x0, max_iter=10000, tol=1e-4)
        """bfgs with backtracking line search"""
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

    function sr1(f, g, x0, max_iter=1000, tol=1e-4, r=1e-8)
        """sr1 with backtracking line search"""
        xk = x0
        H = I
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
