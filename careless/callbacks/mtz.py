
class MTZSaver():
    def __init__(self, data_manager, save_frequency, output_base, inputs=None):
        self.inputs = inputs
        self.dm = data_manager
        self.output_base = output_base
        self.save_frequency = save_frequency

    def write_mtz(self, model, inputs, output_base):
        for i,ds in enumerate(self.dm.get_results(model.surrogate_posterior, inputs)):
            filename = output_base + f'_{i}.mtz'
            ds.write_mtz(filename)

    def __call__(self, iteration, model):
        if iteration % self.save_frequency == 0 and iteration>0:
            output_base = self.output_base + f'_{iteration}'
            self.write_mtz(model, self.inputs, output_base)

