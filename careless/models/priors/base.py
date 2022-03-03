
class Prior():
    """Base class for prior distributions on merged normalized structure factor amplitudes."""
    def log_prob(self):
        raise NotImplementedError("No log_prob method defined. All Priors must implement a log_prob method")


