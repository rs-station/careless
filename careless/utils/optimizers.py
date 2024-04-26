import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from keras.optimizers import Optimizer

# isort: off
from tensorflow.python.util.tf_export import keras_export

class WAdam(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta=0.9,
        epsilon=1e-12,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="WAdam",
        **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta = beta
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 2 types of variables: momentums and velocities.

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta = tf.cast(self.beta, variable.dtype)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        beta_power = tf.pow(tf.cast(self.beta, variable.dtype), local_step)
        alpha = lr / (1 - beta_power)

        if isinstance(gradient, tf.IndexedSlices):
            raise NotImplementedError("No sparse gradient implementation yet!")
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            g = gradient
            delta = g - m
            m.assign_add((1. - beta) * delta)
            v.assign_add((1. - beta) * (beta * delta * (g - m) - v))
            #variable.assign_sub((m * alpha) / (tf.sqrt(v + m*m) + self.epsilon))
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta": self.beta,
                "epsilon": self.epsilon,
            }
        )
        return config


