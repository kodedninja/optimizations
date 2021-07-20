module Utils
    const roundoff_e = eps()^(1/4)

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

     function approx_gradient(f, x; e=roundoff_e)
		"""
		approx_gradient
		
		approximates the gradient of objective function f
		"""
        n = length(x)
        g = zeros(n)
        e_i = zeros(n)
        for i in 1:n
            e_i[i] = 1.
            g[i] = (f(x + e_i * e) - f(x - e_i * e))/2e
            e_i[i] = 0.
        end
        return g
    end

    function approx_hessian(f, x; e=roundoff_e)
		"""
		approx_hessian
		
		approximates the hessian of objective function f
		"""
        n = length(x)
        h = zeros(n, n)
        e_i = zeros(n)
        e_j = zeros(n)

        for i in 1:n
            e_i[i] = 1.
            for j in 1:n
                e_j[j] = 1.
                h[i, j] = (f(x + e * e_i + e * e_j) - f(x + e * e_i) - f(x + e * e_j) + f(x))/e^2
                e_j[j] = 0.
            end
            e_i[i] = 0.
        end
				
        # reshape output when function is not multivariate
        if n == 1
            h = [h[1][1]]
        end

        return h
    end
end
