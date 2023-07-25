from careless.stats import history
from tempfile import TemporaryDirectory
from os.path import exists
import pandas as pd
import pytest


def test_history(history_csv):
    assert exists(history_csv)
    tf = TemporaryDirectory()

    png = f"{tf.name}/out.png"
    assert not exists(png)
    command = f"{history_csv} -o {png}"
    parser = history.ArgumentParser().parse_args(command.split())
    history.run_analysis(parser)
    assert exists(png)

    svg = f"{tf.name}/out.svg"
    assert not exists(svg)
    command = f"{history_csv} -o {svg}"
    parser = history.ArgumentParser().parse_args(command.split())
    history.run_analysis(parser)
    assert exists(svg)

