
class Likelihood():
    def log_prob(self, *args):
        raise NotImplementedError("Likelihood classes must implement a log_prob method")

    def prob(self, *args):
        raise NotImplementedError("Likelihood classes must implement a prob method")

