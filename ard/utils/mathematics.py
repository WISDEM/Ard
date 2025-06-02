import numpy as np
import jax.numpy as jnp
import jax


def smooth_max(x: jnp.ndarray, s: float = 1000.0) -> float:
    """Non-overflowing version of Smooth Max function (see ref 3 and 4 below).
    Calculates the smoothmax (a.k.a. softmax or LogSumExponential) of the elements in x.

    Based on implementation in BYU FLOW Lab's FLOWFarm software at
    (1) https://github.com/byuflowlab/FLOWFarm.jl/tree/master
    which is based on John D. Cook's writings at
    (2) https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    and
    (3) https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
    And based on article in FeedlyBlog
    (4) https://blog.feedly.com/tricks-of-the-trade-logsumexp/

    Args:
        x (list): list of values to be compared
        s (float, optional): alpha for smooth max function. Defaults to 1000.0.
            Larger values of `s` lead to more accurate results, but reduce the smoothness
            of the function.

    Returns:
        float: the smooth max of the provided `x` list
    """

    # get the maximum value and the index of maximum value
    max_ind = jnp.argmax(x)
    max_val = x[max_ind]

    # LogSumExp with smoothing factor s
    exponential = jnp.exp(
        s * (jnp.delete(x, max_ind, assume_unique_indices=True) - max_val)
    )
    r = (jnp.log(1.0 + jnp.sum(exponential)) + s * max_val) / s

    return r


smooth_max = jax.jit(smooth_max)


def smooth_min(x: np.ndarray, s: float = 1000.0) -> float:
    """Finds smooth min using the `smooth_max` function

    Args:
        x (list): list of values to be compared
        s (float, optional): alpha for smooth min function. Defaults to 1000.0.
            Larger values of `s` lead to more accurate results, but reduce the smoothness
            of the function.

    Returns:
        float: the smooth min of the provided `x` list
    """

    return -smooth_max(x=-x, s=s)


smooth_min = jax.jit(smooth_min)


def smooth_norm(vec: np.ndarray, buf: float = 1e-12) -> float:
    """Smooth version of the Frobenius, or 2, norm. This version is nearly equivalent to the 2-norm with the
    maximum absolute error corresponding to the order of the buffer value. The maximum error in the gradient is near unity, but
    the error in the gradient is generally about twice the error in the absolute value. The key benefit of the smooth norm is
    that it is differentiable at 0.0, while the standard norm is undefined at 0.0.

    Args:
        vec (np.ndarray): input vector to be normed
        buf (float, optional): buffer value included in the sum of squares part of the norm. Defaults to 1E-12.

    Returns:
        (float): normed result
    """
    return jnp.sqrt(buf**2 + jnp.sum(vec**2))


smooth_norm = jax.jit(smooth_norm)
smooth_norm_vec = jax.jit(jax.vmap(smooth_norm))
