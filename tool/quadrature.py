import jax
from jax import device_put
import jax.numpy as jnp
import jax.numpy.linalg as linalg
import numpy as np
jax.config.update("jax_enable_x64", True)


class GaussLegendrePiecewise():
    
    def __init__(self, npts):
        """ The cartesian Gauss-Lengendre quadrature information on the reference 1d interval [-1,1].
            INPUT:
                npts:  the number of quadrature points, where (n+1) points gets a 
                        (2*n+1) algebraic precision.
            Device: cuda/cpu
        """
        index = npts - 1
        
        if index == 0:
            self.quadpts = jnp.array([[0.]], dtype=jnp.float64)
            self.weights = jnp.array([[2.]], dtype=jnp.float64)

        else:
            h1 = jnp.linspace(0,index,index+1).astype(jnp.float64)
            h2 = jnp.linspace(0,index,index+1).astype(jnp.float64) * 2

            J = 2*(h1[1:index+1]**2) / (h2[0:index]+2) * \
                jnp.sqrt(1/(h2[0:index]+1)/(h2[0:index]+3))
            J = jnp.diag(J,1) + jnp.diag(J,-1)
            D, V = linalg.eig(J)

            self.quadpts = D.real
            self.weights = (2*V[0,:]**2).real
            self.quadpts = self.quadpts.reshape(D.shape[0],1)
            self.weights = self.weights.reshape(D.shape[0],1) / 2
            
            
    def _seperate(self, pts, weis, K):
        division = pts.shape[0] / K
        partition = [0]+[round(division * (i + 1)) for i in range(K)]
        _pts = [pts[lb:rb,...] for lb, rb in zip(partition[:-1], partition[1:])]
        _weis = [weis[lb:rb,...] for lb, rb in zip(partition[:-1], partition[1:])]
        return _pts, _weis
        
            
    def interval_quadpts(self, interval, h, K=1):
        """ The Gauss-Lengendre quadrature information on a discretized mesh of 1d interval [a,b].
            Usually the mesh is uniform.
            INPUT:
                interval: jnp.array object
                       h: jnp.array object, mesh size 
            OUTPUT: integrator handle, with
                 quadpts: npts-by-1
                 weights: npts-by-1
                       h:  shape=[1] 
            Examples
            -------
            interval = jnp.array([[0, 1]], dtype=jnp.float64)
            h = jnp.array([1/100], dtype=jnp.float64)
        """
        N = (interval[0][1] - interval[0][0])/h[0] + 1
        N = int(N)
        xp = jnp.linspace(interval[0][0], interval[0][1], N).reshape(1,N)
        xp_l = xp[0][0:-1].reshape(1,N-1)
        xp_r = xp[0][1:].reshape(1,N-1)
        quadpts = (self.quadpts*h + xp_l + xp_r) / 2
        weights = jnp.tile(self.weights, quadpts.shape[1])
        quadpts = quadpts.flatten().reshape(-1,1)
        weights = weights.flatten().reshape(-1,1)
        quad_info = self._seperate(quadpts, weights, K)
        
        def integrator(f):
            int_val = 0
            area = jnp.prod(h)
            _pts, _weis = quad_info
            for pts, weis in zip(_pts, _weis):
                f_val = f(pts)
                size = [len(weis)]
                for i in range(len(f_val.shape)-1):
                    size.append(1)
                wei = jnp.reshape(weis, size)
                f_val *= wei * area
                int_val += jnp.sum(f_val, axis=0)
            return int_val
        
        return integrator
    
    
    def rectangle_quadpts(self, rectangle, h, K=1):
        """ The Gauss-Lengendre quadrature information on a discretized mesh of 2d rectangle [a,b]*[c,d].
            Usually the mesh is uniform.
            INPUT:
                interval: np.array object
                       h: np.array object, mesh sizes
            OUTPUT:
                 quadpts: npts-by-2
                 weights: npts-by-1
                       h:  shape=[2]  
            Examples
            -------
            rectangle = np.array([[0, 1], [0, 1]], dtype=np.float64)
            h = np.array([0.01, 0.01], dtype=np.float64)
        """
        
        Nx = (rectangle[0][1] - rectangle[0][0])/h[0] + 1
        Ny = (rectangle[1][1] - rectangle[1][0])/h[1] + 1
        Nx = int(Nx)
        Ny = int(Ny)
        
        xp = jnp.linspace(rectangle[0][0], rectangle[0][1], Nx).reshape(1,Nx)
        yp = jnp.linspace(rectangle[1][0], rectangle[1][1], Ny).reshape(1,Ny)
        xp_l = xp[0][0:-1].reshape(1,Nx-1)
        yp_l = yp[0][0:-1].reshape(1,Ny-1)
        xp_r = xp[0][1:].reshape(1,Nx-1)
        yp_r = yp[0][1:].reshape(1,Ny-1)
        xp = (self.quadpts*h[0] + xp_l + xp_r) / 2
        yp = (self.quadpts*h[1] + yp_l + yp_r) / 2
        
        xpt, ypt = jnp.meshgrid(xp.flatten(), yp.flatten())
        xpt = xpt.flatten().reshape(-1,1)
        ypt = ypt.flatten().reshape(-1,1)
        quadpts = jnp.concatenate((xpt,ypt), axis=1)
        
        weights_x = jnp.tile(self.weights, xp.shape[1])
        weights_y = jnp.tile(self.weights, yp.shape[1])
        weights_x, weights_y = jnp.meshgrid(weights_x.flatten(), weights_y.flatten())
        
        weights_x = weights_x.flatten().reshape(-1,1)
        weights_y = weights_y.flatten().reshape(-1,1)
        weights = weights_x * weights_y
            
        quad_info = self._seperate(quadpts, weights, K)
        
        def integrator(f):
            int_val = 0
            area = jnp.prod(h)
            _pts, _weis = quad_info
            for pts, weis in zip(_pts, _weis):
                f_val = f(pts)
                size = [len(weis)]
                for i in range(len(f_val.shape)-1):
                    size.append(1)
                wei = jnp.reshape(weis, size)
                f_val *= wei * area
                int_val += jnp.sum(f_val, axis=0)
            return int_val

        return integrator
    
    
    
    
class QuasiMonteCarlo():
    
    def __init__(self):
        """ The Quasi-Monte-Carlo sampling information on rectangle domains in any dimension.
            Generates a deterministic set of samples by using the Halton sequence. If you have 
            installed scipy package, try calling: 
            
                        qmc = QuasiMonteCarloQuadrature(device)
                        samples = qmc.n_rectangle_samples(n_rectangle, number_of_samples).
                        
            Otherwise there is another version (much slower when large amount):
            
                        samples = qmc._n_rectangle_samples(n_rectangle, number_of_samples).
            
        Args:
                None
        """     

        
    def _prime(self, index):
        
        prime = jnp.array([2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,  43, 
                          47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101, 103, 107, 
                          109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 
                          191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
                          269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 
                          353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 
                          439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 
                          523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 
                          617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 
                          709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 
                          811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 
                          907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])
        
        assert index <= prime.size, "Only support primes within 1000."
        return prime[index]
    
    
    def _seperate(self, pts, weis, K):
        division = pts.shape[0] / K
        partition = [0]+[round(division * (i + 1)) for i in range(K)]
        _pts = [pts[lb:rb,...] for lb, rb in zip(partition[:-1], partition[1:])]
        _weis = [weis[lb:rb,...] for lb, rb in zip(partition[:-1], partition[1:])]
        return _pts, _weis
        
        
    def _get_standard_halton(self, number_of_samples, prime_base):
        """ Halton sequence on the standard domain [0,1].
        
        Args:
                number_of_samples: the length of Halton sequence
                prime_base: the prime number by which [0,1] will be divided
        """
        
        sequence  = jnp.zeros(number_of_samples)
        number_of_bits = int(1 + jnp.ceil(jnp.log(number_of_samples)/jnp.log(prime_base)))
        
        frac_base = prime_base ** (-(jnp.linspace(1,number_of_bits,number_of_bits)))
        working_base = jnp.zeros(number_of_bits)
        
        for i in range(number_of_samples):
            j = 0
            condition = True
            while condition:
                working_base[j] += 1
                if working_base[j] < prime_base:
                    condition = False
                else:
                    working_base[j] = 0
                    j += 1 
            sequence[i] = jnp.dot(working_base, frac_base)
            
        return sequence
        
        
    def _n_rectangle_samples(self, n_rectangle, number_of_samples, K=1):
        """ The Quasi-Monte-Carlo method supports generating samples in any dimension
            but only in domains with rectangle shapes. Standard sequence of samples will
            be generated in [0,1]^n with n the dimensionality. However, scaling process
            is allowed in this function.

        Args:
            n_rectangle (np.array): rectangle in any dimension
            number_of_samples (int): total number of samples
        """
        
        # dimensionality 
        dim = n_rectangle.shape[0]
        
        # get scaled qmc sequence 
        lengths = jnp.zeros(dim)
        quadpts = jnp.zeros((number_of_samples, dim))
        for i in range(dim):
            prime = self._prime(i)
            length = n_rectangle[i][1] - n_rectangle[i][0]
            sequence = self._get_standard_halton(number_of_samples, prime)
            quadpts[...,i] = n_rectangle[i][0] + length * sequence
            lengths = lengths.at[i].set(length)
        
        # get volumn (measure) of domain
        measure = lengths.prod()
        weights = measure * jnp.ones_like(quadpts).astype(jnp.float64) 
        h = jnp.array([1 / number_of_samples])
        
        # seperate the quadrature into segments
        quad_info = self._seperate(quadpts, weights, K)
        
        def integrator(f):
            int_val = 0
            area = jnp.prod(h)
            _pts, _weis = quad_info
            for pts, weis in zip(_pts, _weis):
                f_val = f(pts)
                size = [len(weis)]
                for i in range(len(f_val.shape)-1):
                    size.append(1)
                wei = jnp.reshape(weis, size)
                f_val *= wei * area
                int_val += jnp.sum(f_val, axis=0)
            return int_val
        
        return integrator
            
    
    def n_rectangle_samples(self, n_rectangle, number_of_samples, K=1):
        
        # needs scipy.stats.qmc
        from scipy.stats import qmc 
        
        # dimensionality
        dim = n_rectangle.shape[0]
        
        # get scaled qmc sequence 
        sampler = qmc.Halton(d=dim, scramble=False, seed=10)
        sample = sampler.random(n=number_of_samples)
        quadpts = qmc.scale(sample, n_rectangle[...,0], n_rectangle[...,1])
        
        # get volumn (measure) of domain
        lengths = jnp.zeros(dim)
        for i in range(dim):
            length = n_rectangle[i][1] - n_rectangle[i][0]
            lengths = lengths.at[i].set(length)
        measure = lengths.prod()
        weights = measure * jnp.ones((quadpts.shape[0],1)).astype(jnp.float64)  
        h = jnp.array([1 / number_of_samples])
        
        # seperate the quadrature into segments
        quad_info = self._seperate(quadpts, weights, K)
        
        def integrator(f):
            int_val = 0
            area = jnp.prod(h)
            _pts, _weis = quad_info
            for pts, weis in zip(_pts, _weis):
                f_val = f(pts)
                size = [len(weis)]
                for i in range(len(f_val.shape)-1):
                    size.append(1)
                wei = jnp.reshape(weis, size)
                f_val *= wei * area
                int_val += jnp.sum(f_val, axis=0)
            return int_val
        
        return integrator 
    
    
    
class MonteCarlo():
    
    def __init__(self):
        """ The Monte-Carlo sampling information on rectangle domains in any dimension.
            Generates a random set of samples.
            
        Args:
                None
        """  
 
        
    def _seperate(self, pts, weis, K):
        division = pts.shape[0] / K
        partition = [0]+[round(division * (i + 1)) for i in range(K)]
        _pts = [pts[lb:rb,...] for lb, rb in zip(partition[:-1], partition[1:])]
        _weis = [weis[lb:rb,...] for lb, rb in zip(partition[:-1], partition[1:])]
        return _pts, _weis
    
        
    def n_rectangle_samples(self, n_rectangle, number_of_samples, K=1):
        
        # dimensionality
        dim = n_rectangle.shape[0]
        
        # get scaled mc sequence 
        starts = n_rectangle[...,0]
        widths = n_rectangle[...,1] - n_rectangle[...,0]
        pts = np.random.rand(number_of_samples, dim)
        quadpts = device_put(pts*widths + starts)
        
        # get volumn (measure) of domain
        lengths = jnp.zeros(dim)
        for i in range(dim):
            length = n_rectangle[i][1] - n_rectangle[i][0]
            lengths = lengths.at[i].set(length)
        measure = lengths.prod()
        weights = measure * jnp.ones((quadpts.shape[0],1)).astype(jnp.float64)  
        h = jnp.array([1 / number_of_samples])
        
        # seperate the quadrature into segments
        quad_info = self._seperate(quadpts, weights, K)
        
        def integrator(f):
            int_val = 0
            area = jnp.prod(h)
            _pts, _weis = quad_info
            for pts, weis in zip(_pts, _weis):
                f_val = f(pts)
                size = [len(weis)]
                for i in range(len(f_val.shape)-1):
                    size.append(1)
                wei = jnp.reshape(weis, size)
                f_val *= wei * area
                int_val += jnp.sum(f_val, axis=0)
            return int_val
        
        return integrator 