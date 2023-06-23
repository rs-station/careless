from careless.stats import completeness
from tempfile import TemporaryDirectory
from os.path import exists
import pandas as pd
import pytest



@pytest.mark.parametrize("bins", [1, 10])
def test_completeness(merged_mtz, bins):
    tf = TemporaryDirectory()
    csv = f"{tf.name}/out.csv"
    png = f"{tf.name}/out.png"

    command = f"-o {csv} -i {png} -b {bins} {merged_mtz}"

    parser = completeness.ArgumentParser().parse_args(command.split())

    assert not exists(csv)
    assert not exists(png)
    completeness.run_analysis(parser)
    assert exists(csv)
    assert exists(png)

    df = pd.read_csv(csv)
    assert len(df) == bins + 2

