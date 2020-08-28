class QuadratureBase():
    def expected_log_likelihood(self):
        raise NotImplementedError("Extensions of this class must implement an expected_log_likelihood method")
