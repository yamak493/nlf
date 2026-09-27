"""NLF の推定結果（SMPL パラメータ）を MMD 用の VMD モーションに変換する（仕様: vmd.md）。

    from nlf2vmd import convert
    result = convert('motion.npz', 'motion.vmd', pmx='model.pmx')

コマンドラインからは python -m nlf2vmd --help。
"""
from .body_model import BodyModel
from .config import dump_config, load_config
from .pipeline import ConversionResult, convert
from .variants import VARIANTS, write_variant

__all__ = ['BodyModel', 'ConversionResult', 'VARIANTS', 'convert', 'dump_config', 'load_config',
           'write_variant']
