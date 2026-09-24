import numpy as np
import pytest

from nlf2vmd import convert, load_config
from nlf2vmd.synthetic import synthetic_body_model, synthetic_walk


@pytest.fixture(scope='session')
def body_model():
    return synthetic_body_model()


@pytest.fixture
def run_walk(tmp_path, body_model):
    """合成の歩行データを変換して ConversionResult を返す。"""
    def run(overrides=(), name='walk.vmd', **walk_kwargs):
        motion = synthetic_walk(**walk_kwargs)
        cfg = load_config(overrides=['diagnostics.enabled=false', *overrides])
        result = convert(motion, tmp_path / name, body_model=body_model, config=cfg, log=None)
        return motion, result
    return run


def rotation_angles(q):
    """連続するフレーム間の回転角 [deg]。q: (T, ..., 4)。"""
    from nlf2vmd import quat
    return np.rad2deg(quat.angle_between(q[:-1], q[1:]))
