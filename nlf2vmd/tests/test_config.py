import pytest

from nlf2vmd.config import dump_config, load_config


def test_defaults_and_overrides(tmp_path):
    cfg = load_config(overrides=['center.mode=B', 'jitter.one_euro.groups.arm.beta=2'])
    assert cfg.center.mode == 'B'
    assert cfg.jitter.one_euro.groups.arm.beta == 2
    assert cfg.contact.enter_height_m == 0.03          # 触っていない値は既定のまま

    path = tmp_path / 'c.yaml'
    path.write_text('foot_ik:\n  blend_frames: 8\n', encoding='utf-8')
    assert load_config(path).foot_ik.blend_frames == 8

    dump_config(cfg, tmp_path / 'dump.yaml')
    assert load_config(tmp_path / 'dump.yaml').center.mode == 'B'


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        load_config({'center': {'mdoe': 'B'}})
    with pytest.raises(ValueError):
        load_config(overrides=['center.mode'])
