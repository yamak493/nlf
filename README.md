# Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation
* [NeurIPS'24 paper](https://arxiv.org/abs/2407.07532) by István Sárándi and Gerard Pons-Moll
* [Project page](https://istvansarandi.com/nlf)

Models for PyTorch and TensorFlow are available for noncommercial research use under Releases, and usage examples are given in `demo.ipynb`. Stay tuned for more detailed docs.

Training code is provided for both PyTorch and TensorFlow.

## Colab デモ（日本語）: mp4 → モーション抽出 → 踊るマネキン動画

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamak493/nlf/blob/claude/kind-mendel-tdveor/mp4_to_mannequin_ja.ipynb)

`mp4_to_mannequin_ja.ipynb` は、入力の mp4 を URL（既定: `https://made-by-free.com/night-fire.mp4`）からダウンロードし、始点秒数〜終点秒数を指定するだけで

1. NLF v0.3.2 の学習済みモデル（自動ダウンロード）でモーションを抽出し（`motion.npz`）、
2. そのモーションを反映したマネキンをレンダリングして、
3. 元動画の音声付き mp4 として書き出し、
4. モーションを FBX（`motion.fbx`：スケルトン＋スキン付きマネキン＋アニメーション）でも書き出す

という一連の処理を行う日本語ノートブックです。SMPL 公式ファイルが無くても動作します
（[SMPLFitter](https://github.com/isarandi/smplfitter) は NLF の内部でも使われています）。

## Acknowledgments
This work was supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039A. This work is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 409792180 (Emmy Noether Programme, project: Real Virtual Humans). GPM is a member of the Machine Learning Cluster of Excellence, EXC number 2064/1 –Project number 390727645. The project was made possible by funding from the Carl Zeiss Foundation.

## BibTeX
```
@article{sarandi2024nlf,
    title     = {Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation},
    author    = {S\'ar\'andi, Istv\'an and Pons-Moll, Gerard},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2024}
}
```
