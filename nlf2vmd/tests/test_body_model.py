"""体モデル: 根の親が -1 以外の値で入っていても（NLF の TorchScript など）正しく読めること。"""
import numpy as np
import pytest

from nlf2vmd import BodyModel, convert, load_config
from nlf2vmd.body_model import SMPL_PARENTS, sanitize_parents
from nlf2vmd.synthetic import synthetic_body_model, synthetic_walk

# -1 を符号なし 32 ビットで表した値と、それが float32 で丸められた値
UNSIGNED_ROOTS = [4294967295, 4294967296.0]


@pytest.mark.parametrize('root', UNSIGNED_ROOTS)
def test_unsigned_root_parent_is_treated_as_root(root):
    parents = SMPL_PARENTS.astype(np.float64)
    parents[0] = root
    np.testing.assert_array_equal(sanitize_parents(parents, 24), SMPL_PARENTS)


def test_broken_tree_falls_back_to_smpl():
    parents = SMPL_PARENTS.copy()
    parents[5] = 30
    with pytest.warns(UserWarning):
        np.testing.assert_array_equal(sanitize_parents(parents, 24), SMPL_PARENTS)


def test_from_torch_keeps_integer_buffers(tmp_path):
    torch = pytest.importorskip('torch')
    bm = synthetic_body_model()

    class TorchBodyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for k in ('v_template', 'shapedirs', 'J_template', 'J_shapedirs', 'weights'):
                self.register_buffer(k, torch.tensor(getattr(bm, k), dtype=torch.float32))
            parents = torch.tensor(SMPL_PARENTS)
            parents[0] = 4294967295
            self.register_buffer('kintree_parents_tensor', parents)

    model = BodyModel.from_torch(TorchBodyModel())
    np.testing.assert_array_equal(model.parents, SMPL_PARENTS)

    # 書き出した npz から読み直しても変換まで通る
    model.save_npz(tmp_path / 'smpl_body_model.npz')
    cfg = load_config(overrides=['diagnostics.enabled=false'])
    convert(synthetic_walk(num_frames=60), tmp_path / 'w.vmd',
            body_model=tmp_path / 'smpl_body_model.npz', config=cfg, log=None)
