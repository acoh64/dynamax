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

# class RealToTracelessBijector(tfb.Bijector):
#     """Bijector that maps unconstrained real vectors to traceless matrices.
    
#     For an n×n matrix, takes n²-1 parameters and constructs a traceless matrix.
#     For 3×3, the 8 parameters map to a matrix where the last diagonal element
#     is determined by ensuring the trace is zero.
#     """
    
#     def __init__(self,
#                  matrix_size,
#                  validate_args=False,
#                  name="real_to_traceless"):
#         super().__init__(
#             forward_min_event_ndims=1,
#             inverse_min_event_ndims=2,
#             validate_args=validate_args,
#             name=name)
#         self.matrix_size = matrix_size
        
#     def _forward(self, x):
#         # For n×n matrix, x should have n²-1 elements
#         n = self.matrix_size
#         if x.shape[-1] != n * n - 1:
#             raise ValueError(f"Input should have {n * n - 1} elements for {n}x{n} matrix. {x.shape}")
            
#         # First n²-n elements are off-diagonal entries (row by row)
#         # Last n-1 elements are first n-1 diagonal entries
#         off_diag = x[:(n * n - n)]
#         diag_elements = x[(n * n - n):]
        
#         # Create the matrix
#         matrix = jnp.zeros((n, n))
        
#         # Fill off-diagonal elements
#         idx = 0
#         for i in range(n):
#             for j in range(n):
#                 if i != j:
#                     matrix = matrix.at[i, j].set(off_diag[idx])
#                     idx += 1
        
#         # Fill first n-1 diagonal elements
#         matrix = matrix.at[jnp.arange(n-1), jnp.arange(n-1)].set(diag_elements)
        
#         # Set last diagonal element to ensure trace is zero
#         last_diag = n - jnp.sum(diag_elements)
#         matrix = matrix.at[n-1, n-1].set(last_diag)
        
#         return matrix
    
#     def _inverse(self, y):
#         n = self.matrix_size
        
#         # Extract off-diagonal elements
#         mask = ~jnp.eye(n, dtype=bool)
#         off_diag = y[mask]
        
#         # Extract first n-1 diagonal elements
#         diag_elements = jnp.diagonal(y)[:-1]
        
#         # Concatenate parameters
#         return jnp.concatenate([off_diag, diag_elements])

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