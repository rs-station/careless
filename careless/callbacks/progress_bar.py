from tqdm.keras import TqdmCallback
from tqdm.auto import tqdm

class auto_tqdm(tqdm):
    def __init__(self, *args, numeric_format='{0:.3e}', **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_format=numeric_format

    def format_num(self, n):
        return self.numeric_format.format(n)

class ProgressBar(TqdmCallback):
    """ Keras Callback based on tqdm.keras.TqdmCallback but with fixed width metrics. """
    def __init__(self, *args, tqdm_class=auto_tqdm, verbose=0, **kwargs):
        super().__init__(*args, tqdm_class=auto_tqdm, verbose=verbose, **kwargs)

