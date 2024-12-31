import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
import jax
from scipy.optimize import root
import numpy as np
# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
class PSDToRealBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            tfb.Invert(tfb.FillTriangular()),
            tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
            tfb.Invert(tfb.CholeskyOuterProduct()),
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)


class RealToPSDBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            tfb.CholeskyOuterProduct(),
            tfb.TransformDiagonal(tfb.Exp()),
            tfb.FillTriangular(),
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)

class RealToTracelessBijector(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to traceless matrices.
    
    For an n×n matrix, takes n²-1 parameters and constructs a traceless matrix.
    For 3×3, the 8 parameters map to a matrix where the last diagonal element
    is determined by ensuring the trace is zero.
    
    This bijector operates over the first dimension (num_states) of the input,
    where each state has its own traceless matrix.
    """
    
    def __init__(self,
                 matrix_size,
                 validate_args=False,
                 name="real_to_traceless"):
        super().__init__(
            forward_min_event_ndims=1,  # Input is a vector of parameters
            inverse_min_event_ndims=2,  # Output is a matrix
            validate_args=validate_args,
            name=name)
        self.matrix_size = matrix_size
        
    def _forward(self, x):
        # x shape: (num_states, n²-1)
        n = self.matrix_size
        if x.shape[-1] != n * n - 1:
            raise ValueError(f"Input should have {n * n - 1} elements for {n}x{n} matrix. Got shape {x.shape}")
            
        # Split parameters for each state
        off_diag = x[..., :(n * n - n)]  # (..., n²-n)
        diag_elements = x[..., (n * n - n):]  # (..., n-1)
        
        # Create batch of matrices
        matrix = jnp.zeros(x.shape[:-1] + (n, n))  # (..., n, n)
        
        # Fill off-diagonal elements
        idx = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix = matrix.at[..., i, j].set(off_diag[..., idx])
                    idx += 1
        
        # Fill first n-1 diagonal elements
        for i in range(n-1):
            matrix = matrix.at[..., i, i].set(diag_elements[..., i])
        
        # Set last diagonal element to ensure trace is zero
        last_diag = self.matrix_size - jnp.sum(diag_elements, axis=-1)
        matrix = matrix.at[..., n-1, n-1].set(last_diag)
        
        return matrix
    
    def _inverse(self, y):
        # y shape: (num_states, n, n)
        n = self.matrix_size
        
        # Extract off-diagonal elements
        mask = ~jnp.eye(n, dtype=bool)
        off_diag = y[..., mask]  # (..., n²-n)
        
        # Extract first n-1 diagonal elements
        diag_elements = jnp.diagonal(y, axis1=-2, axis2=-1)[..., :-1]  # (..., n-1)
        
        # Concatenate parameters
        return jnp.concatenate([off_diag, diag_elements], axis=-1)  # (..., n²-1)
    

class DiagonalBijector(tfb.Bijector):
    """Bijector that maps vectors to diagonal matrices.
    
    Takes inputs of shape (num_states, n) and returns matrices of shape (num_states, n, n)
    where n is the size of each diagonal matrix.
    """
    
    def __init__(self,
                 validate_args=False,
                 name="diagonal"):
        super().__init__(
            forward_min_event_ndims=1,  # Input is a vector
            inverse_min_event_ndims=2,  # Output is a matrix
            validate_args=validate_args,
            name=name)
    
    def _forward(self, x):
        # x shape: (num_states, n)
        n = x.shape[-1]
        # Create batch of diagonal matrices
        return jnp.einsum('...i,ij->...ij', x, jnp.eye(n))
    
    def _inverse(self, y):
        # y shape: (num_states, n, n)
        return jnp.diagonal(y, axis1=-2, axis2=-1)

class RealToPSDDiagonalBijector(tfb.Chain):
    """Bijector that maps unconstrained real vectors to positive semidefinite diagonal matrices.
    
    This bijector simply applies the exponential function to ensure diagonal elements are positive,
    then places these values on the diagonal of a matrix with zeros elsewhere.
    """

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            DiagonalBijector(),  # Creates a diagonal matrix
            tfb.Exp(),  # Makes elements positive
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)

class NambuBijector4D(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to Nambu dynamics."""
    def __init__(self, name=None, validate_args=False):
        super().__init__(name=name, validate_args=validate_args, forward_min_event_ndims=1, inverse_min_event_ndims=2)

    def _forward(self, x):
        return jax.vmap(self._tmp_forward, in_axes=(0,))(x)

    def _inverse(self, y):
        # return jax.vmap(self._tmp_inverse, in_axes=(0,))(y)
        results = []
        for element in y:
            print(element)
            print("here")
            result = self._tmp_inverse(element)
            results.append(result)
        print("done")
        return jnp.array(results)

    def _tmp_forward(self, x):
       
        h1_1, h1_2, h1_3, h1_4 = x[:4]
        h2_1, h2_2, h2_3, h2_4 = x[4:8]
        h3_1, h3_2, h3_3, h3_4 = x[8:]
        m12 = 2.0*h1_2 * (h2_3*h3_4 - h2_4*h3_3)
        m13 = 2.0*h1_3 * (h2_4*h3_2 - h2_2*h3_4)
        m14 = 2.0*h1_4 * (h2_2*h3_3 - h2_3*h3_2)

        m21 = 2.0*h1_1 * (h2_4*h3_3 - h2_3*h3_4)
        m23 = 2.0*h1_3 * (h2_1*h3_4 - h2_4*h3_1)
        m24 = 2.0*h1_4 * (h2_3*h3_1 - h2_1*h3_3)

        m31 = 2.0*h1_1 * (h2_2*h3_4 - h2_4*h3_2)
        m32 = 2.0*h1_2 * (h2_4*h3_1  - h2_1*h3_4)
        m34 = 2.0*h1_4 * (h2_1*h3_2 - h2_2*h3_1)

        m41 = 2.0*h1_1 * (h2_3*h3_2 - h2_2*h3_3)
        m42 = 2.0*h1_2 * (h2_1*h3_3 - h2_3*h3_1)
        m43 = 2.0*h1_3 * (h2_2*h3_1 - h2_1*h3_2)

        # need to put 1.0 along diagonal to account for time discretization
        return jnp.array([[1.0, m12, m13, m14], [m21, 1.0, m23, m24], [m31, m32, 1.0, m34], [m41, m42, m43, 1.0]])
    
    def _equations(self, params, x):
        h1_1, h1_2, h1_3, h1_4, h2_1, h2_2, h2_3, h2_4, h3_1, h3_2, h3_3, h3_4 = params
        M = x
        m12 = M[0, 1]
        m13 = M[0, 2]
        m14 = M[0, 3]
        m21 = M[1, 0]
        m23 = M[1, 2]
        m24 = M[1, 3]
        m31 = M[2, 0]
        m32 = M[2, 1]
        m34 = M[2, 3]
        m41 = M[3, 0]
        m42 = M[3, 1]
        m43 = M[3, 2]
        
        return jnp.array([
            m12 - (2.0*h1_2 * (h2_3*h3_4 - h2_4*h3_3)),
            m13 - (2.0*h1_3 * (h2_4*h3_2 - h2_2*h3_4)),
            m14 - (2.0*h1_4 * (h2_2*h3_3 - h2_3*h3_2)),
            m21 - (2.0*h1_1 * (h2_4*h3_3 - h2_3*h3_4)),
            m23 - (2.0*h1_3 * (h2_1*h3_4 - h2_4*h3_1)),
            m24 - (2.0*h1_4 * (h2_3*h3_1 - h2_1*h3_3)),
            m31 - (2.0*h1_1 * (h2_2*h3_4 - h2_4*h3_2)),
            m32 - (2.0*h1_2 * (h2_4*h3_1  - h2_1*h3_4)),
            m34 - (2.0*h1_4 * (h2_1*h3_2 - h2_2*h3_1)),
            m41 - (2.0*h1_1 * (h2_3*h3_2 - h2_2*h3_3)),
            m42 - (2.0*h1_2 * (h2_1*h3_3 - h2_3*h3_1)),
            m43 - (2.0*h1_3 * (h2_2*h3_1 - h2_1*h3_2)),
        ])
    
    def _recover_params(self, eq, init_center=0.0, init_scale=1.0, max_iter=20000):
        res = None
        num_tries = 0
        while num_tries < max_iter:
            if num_tries > 0 and num_tries % 2000 == 0:
                init_scale *= 0.1
                print(f"Reducing scale to {init_scale}")
            res = root(eq, init_scale * np.random.randn(12) + init_center)
            if res.success:
                break
            num_tries += 1
        if num_tries == max_iter:
            raise ValueError("Failed to converge")
            # print("Failed to converge")
        return res
    
    def _tmp_inverse(self, x):
        eqs = jax.jit(lambda params: self._equations(params, x))
        res = self._recover_params(eqs)
        return res.x
    
class NambuBijector3D(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to Nambu dynamics."""
    def __init__(self, name=None, validate_args=False):
        super().__init__(name=name, validate_args=validate_args, forward_min_event_ndims=1, inverse_min_event_ndims=2)

    def _forward(self, x):
        return jax.vmap(self._tmp_forward, in_axes=(0,))(x)

    def _inverse(self, y):
        # return jax.vmap(self._tmp_inverse, in_axes=(0,))(y)
        results = []
        for element in y:
            print(element)
            print("here")
            result = self._tmp_inverse(element)
            results.append(result)
        print("done")
        return jnp.array(results)

    def _tmp_forward(self, x):
       
        a, b, c, d, e, f, g, h, k = x

        a = a**2
        b = b**2
        c = c**2
        
        d = 0.0
        e = 0.0
        f = 0.0
        
        m11 = d*k - e*h
        m12 = 2.0*b*k - f*h
        m13 = f*k - 2.0*c*h
        m21 = e*g - 2.0*a*k
        m22 = f*g - d*k
        m23 = 2.0*g*c - e*k
        m31 = 2.0*a*h - d*g
        m32 = d*h - 2.0*b*g
        m33 = e*h - f*g

        # need to put 1.0 along diagonal to account for time discretization
        return jnp.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]) + jnp.eye(3)
    
    def _equations(self, params, x):
        a, b, c, d, e, f, g, h, k = params
        M = x - jnp.eye(3)
        m11 = M[0, 0]
        m12 = M[0, 1]
        m13 = M[0, 2]
        m21 = M[1, 0]
        m22 = M[1, 1]
        m23 = M[1, 2]
        m31 = M[2, 0]
        m32 = M[2, 1]
        m33 = M[2, 2]
        
        return jnp.array([
            m11 - (d*k - e*h),
            m12 - (2.0*b*k - f*h),
            m13 - (f*k - 2.0*c*h),
            m21 - (e*g - 2.0*a*k),
            m22 - (f*g - d*k),
            m23 - (2.0*g*c - e*k),
            m31 - (2.0*a*h - d*g),
            m32 - (d*h - 2.0*b*g),
            m33 - (e*h - f*g)
        ])
    
    def _recover_params(self, eq, init_center=0.0, init_scale=1.0, max_iter=20000):
        res = None
        num_tries = 0
        while num_tries < max_iter:
            if num_tries > 0 and num_tries % 2000 == 0:
                init_scale *= 0.1
                print(f"Reducing scale to {init_scale}")
            res = root(eq, init_scale * np.random.randn(9) + init_center)
            if res.success:
                break
            num_tries += 1
        if num_tries == max_iter:
            raise ValueError("Failed to converge")
            # print("Failed to converge")
        return res
    
    def _tmp_inverse(self, x):
        eqs = jax.jit(lambda params: self._equations(params, x))
        res = self._recover_params(eqs)
        return res.x

class NambuBijector3DSimple(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to Nambu dynamics."""
    def __init__(self, name=None, validate_args=False):
        super().__init__(name=name, validate_args=validate_args, forward_min_event_ndims=1, inverse_min_event_ndims=2)

    def _forward(self, x):
        return jax.vmap(self._tmp_forward, in_axes=(0,))(x)

    def _inverse(self, y):
        # return jax.vmap(self._tmp_inverse, in_axes=(0,))(y)
        results = []
        for element in y:
            print(element)
            print("here")
            result = self._tmp_inverse(element)
            results.append(result)
        print("done")
        return jnp.array(results)

    def _tmp_forward(self, x):
       
        a, b, c, g, h, k = x

        a = a**2
        b = b**2
        c = c**2
        
        m12 = 2.0*b*k
        m13 = -2.0*c*h
        m21 = -2.0*a*k
        m23 = 2.0*g*c
        m31 = 2.0*a*h
        m32 = -2.0*b*g

        # need to put 1.0 along diagonal to account for time discretization
        return jnp.array([[0.0, m12, m13], [m21, 0.0, m23], [m31, m32, 0.0]]) + jnp.eye(3)
    
    def _equations(self, params, x):
        a, b, c, g, h, k = params
        M = x - jnp.eye(3)
        m11 = M[0, 0]
        m12 = M[0, 1]
        m13 = M[0, 2]
        m21 = M[1, 0]
        m22 = M[1, 1]
        m23 = M[1, 2]
        m31 = M[2, 0]
        m32 = M[2, 1]
        m33 = M[2, 2]
        
        return jnp.array([
            m12 - (2.0*b*k),
            m13 - (-2.0*c*h),
            m21 - (-2.0*a*k),
            m23 - (2.0*g*c),
            m31 - (2.0*a*h),
            m32 - (-2.0*b*g),
        ])
    
    def _recover_params(self, eq, init_center=0.0, init_scale=1.0, max_iter=20000):
        res = None
        num_tries = 0
        while num_tries < max_iter:
            if num_tries > 0 and num_tries % 2000 == 0:
                init_scale *= 0.1
                print(f"Reducing scale to {init_scale}")
            res = root(eq, init_scale * np.random.randn(6) + init_center)
            if res.success:
                break
            num_tries += 1
        if num_tries == max_iter:
            raise ValueError("Failed to converge")
            # print("Failed to converge")
        return res
    
    def _tmp_inverse(self, x):
        eqs = jax.jit(lambda params: self._equations(params, x))
        res = self._recover_params(eqs)
        return res.x
    
class NambuBijector3DNoInverse(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to Nambu dynamics."""
    def __init__(self, name=None, validate_args=False):
        super().__init__(name=name, validate_args=validate_args, forward_min_event_ndims=1, inverse_min_event_ndims=2)

    def _forward(self, x):
        return jax.vmap(self._tmp_forward, in_axes=(0,))(x)

    def _tmp_forward(self, x):
       
        a, b, c, d, e, f, g, h, k = x

        a = a**2
        b = b**2
        c = c**2

        d = 0.0
        e = 0.0
        f = 0.0
        
        m11 = d*k - e*h
        m12 = 2.0*b*k - f*h
        m13 = f*k - 2.0*c*h
        m21 = e*g - 2.0*a*k
        m22 = f*g - d*k
        m23 = 2.0*g*c - e*k
        m31 = 2.0*a*h - d*g
        m32 = d*h - 2.0*b*g
        m33 = e*h - f*g

        # need to put 1.0 along diagonal to account for time discretization
        return jnp.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]) + jnp.eye(3)
    
    def _inverse(self, y):
        return jnp.zeros_like(y)

class NambuBijector3DNoInverseScale(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to Nambu dynamics."""
    def __init__(self, params, name=None, validate_args=False):
        self.params = params
        print(self.params.shape)
        print(self.params)
        super().__init__(name=name, validate_args=validate_args, forward_min_event_ndims=1, inverse_min_event_ndims=2)

    def _forward(self, x):
        return jax.vmap(self._tmp_forward, in_axes=(0,))(x)

    def _tmp_forward(self, x):
       
        scale = x

        print("scale", scale.shape)

        a = scale * self.params[0]**2
        b = scale * self.params[1]**2
        c = scale * self.params[2]**2

        d = 0.0
        e = 0.0
        f = 0.0

        g = scale * self.params[6]
        h = scale * self.params[7]
        k = scale * self.params[8]
        
        m11 = d*k - e*h
        m12 = 2.0*b*k - f*h
        m13 = f*k - 2.0*c*h
        m21 = e*g - 2.0*a*k
        m22 = f*g - d*k
        m23 = 2.0*g*c - e*k
        m31 = 2.0*a*h - d*g
        m32 = d*h - 2.0*b*g
        m33 = e*h - f*g

        # need to put 1.0 along diagonal to account for time discretization
        return jnp.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]]) + jnp.eye(3)
    
    def _inverse(self, y):
        return jnp.zeros_like(y)


class NambuBijector2D(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to Nambu dynamics."""
    def __init__(self, name=None, validate_args=False):
        super().__init__(name=name, validate_args=validate_args, forward_min_event_ndims=1, inverse_min_event_ndims=2)

    def _forward(self, x):
        return jax.vmap(self._tmp_forward, in_axes=(0,))(x)

    def _inverse(self, y):
        return jax.vmap(self._tmp_inverse, in_axes=(0,))(y)        

    def _tmp_forward(self, x):
       
        a, b, c = x

        # a = a**2
        # b = b**2
        # a = jnp.exp(a)
        # b = jnp.exp(b)
        # c = 0.0
        
        m12 = 2.0 * b
        m21 = -2.0 * a
        m11 = c
        m22 = -c

        # need to put 1.0 along diagonal to account for time discretization
        return jnp.array([[m11, m12], [m21, m22]]) + jnp.eye(2)
    
    def _tmp_inverse(self, x):
        M = x - jnp.eye(2)
        m12 = M[0, 1]
        m21 = M[1, 0]
        m11 = M[0, 0]
        # m22 = x[1, 1]
        # assert jnp.allclose(m11, -m22)
        # return jnp.array([jnp.sqrt(-m21 / 2.0), jnp.sqrt(m12 / 2.0), m11])
        # return jnp.array([jnp.log(jnp.sqrt(-m21 / 2.0)), jnp.log(jnp.sqrt(m12 / 2.0)), m11])
        return jnp.array([-m21 / 2.0, m12 / 2.0, m11])
        # return jnp.array([jnp.sqrt(-m21 / 2.0), jnp.sqrt(m12 / 2.0), 0.0])