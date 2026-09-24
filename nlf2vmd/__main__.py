"""コマンドライン: python -m nlf2vmd motion.npz -o motion.vmd --pmx model.pmx"""
import argparse
import sys
from pathlib import Path

from .config import dump_config, load_config
from .diagnostics import format_metrics
from .pipeline import convert


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m nlf2vmd',
        description='NLF のモーション（motion.npz）を MMD 用の VMD に変換します（仕様: vmd.md）。')
    ap.add_argument('input', nargs='?', help='入力の npz（pose / betas / trans / fps）')
    ap.add_argument('-o', '--output', help='出力する .vmd（既定: 入力と同じ名前）')
    ap.add_argument('--pmx', help='対象モデルの .pmx（脚長・ボーン位置に使う。省略時は標準ボーン）')
    ap.add_argument('--body-model', help='SMPL 体モデルの npz（省略時は入力と同じフォルダの '
                                         'smpl_body_model.npz）')
    ap.add_argument('--config', help='設定ファイル（YAML / JSON）。書いた項目だけ既定値を上書き')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='設定を 1 項目上書き（例: --set center.mode=B）。複数指定可')
    ap.add_argument('--diag-dir', help='診断出力（JSON・PNG）の保存先（既定: <出力名>_diag）')
    ap.add_argument('--no-plots', action='store_true', help='グラフを出力しない')
    ap.add_argument('--dump-config', metavar='PATH', help='既定値（＋上書き）を YAML に書き出して終了')
    ap.add_argument('--demo', action='store_true',
                    help='合成データ（歩行）と簡易体モデルで変換を試す（入力ファイル不要）')
    args = ap.parse_args(argv)

    overrides = list(args.set)
    if args.no_plots:
        overrides.append('diagnostics.plots=false')
    if args.dump_config:
        dump_config(load_config(args.config, overrides), args.dump_config)
        print('設定を書き出しました:', args.dump_config)
        return 0

    body_model = args.body_model
    if args.demo:
        from .synthetic import synthetic_body_model, synthetic_walk
        source = synthetic_walk(num_frames=240, noise_deg=2.0)
        body_model = synthetic_body_model()
        output = args.output or 'demo_walk.vmd'
    else:
        if not args.input:
            ap.error('入力の npz を指定してください（試すだけなら --demo）')
        source = args.input
        output = args.output or str(Path(args.input).with_suffix('.vmd'))

    result = convert(source, output, pmx=args.pmx, body_model=body_model, config=args.config,
                     overrides=overrides, diag_dir=args.diag_dir)
    if result.metrics:
        print('\n評価指標（処理前 → 処理後）:')
        for line in format_metrics(result.metrics):
            print('  ' + line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
