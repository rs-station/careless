import tensorflow as tf

def softplus_inverse(tensor):
    return tf.math.log(tf.math.exp(tensor) - 1.)
