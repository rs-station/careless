from careless.stats import completeness
from tempfile import TemporaryDirectory
from os.path import exists
import pandas as pd
import pytest



@pytest.mark.parametrize("bins", [1, 10])
def test_completeness(merged_mtz, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    command = f"-o {csv} -b {bins} {merged_mtz}"

    parser = completeness.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    completeness.run_analysis(parser, show=False)
    assert exists(csv)

    df = pd.read_csv(csv)
    assert len(df) == bins + 2

