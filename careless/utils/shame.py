import tensorflow as tf

def sanitize_tensor(tensor, replacement_val=0.):
    """Replace infinite entries `replacement_val`."""
    return tf.where(~tf.math.is_finite(tensor), replacement_val*tf.ones_like(tensor), tensor)

