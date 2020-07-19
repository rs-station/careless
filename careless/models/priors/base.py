
class Prior():
    """Base class for prior distributions on merged normalized structure factor amplitudes."""
    def log_prob(self):
        raise NotImplementedError("No log_prob method defined. All Priors must implement a log_prob method")

    def prob(self):
        raise NotImplementedError("No prob method defined. All Priors must implement a prob method")

    def stddev(self):
        raise NotImplementedError("No stddev method defined. All Priors must implement a stddev method")

    def mean(self):
        raise NotImplementedError("No mean method defined. All Priors must implement a mean method")

