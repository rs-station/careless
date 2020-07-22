class Scaler:
    """ Abstract base class for all scale factor correction models. """
    trainable_variables = None

    def sample(self):
        error_message = """
        Extensions of this class must implement a sample method which accepts at least two parameters. 

        Paramters
        ---------
        return_kl_term : bool
            If true, return a tuple of tensors `(sample, kl_term)`.
        sample_shape : tuple, int
            A shape for the sample to be returned. Default is ().
        """
        raise NotImplementedError(error_message)
