import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp
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

class NumpyroBijector(tfb.Bijector):
    """Bijector that maps unconstrained real vectors to positive semidefinite diagonal matrices."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def _forward(self, x):
        return x

    def _inverse(self, y):
        return y
