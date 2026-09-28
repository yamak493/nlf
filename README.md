# Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation
* [NeurIPS'24 paper](https://arxiv.org/abs/2407.07532) by István Sárándi and Gerard Pons-Moll
* [Project page](https://istvansarandi.com/nlf)

Models for PyTorch and TensorFlow are available for noncommercial research use under Releases, and usage examples are given in `demo.ipynb`. Stay tuned for more detailed docs.

Training code is provided for both PyTorch and TensorFlow.

## Colab デモ（日本語）: mp4 → モーション抽出 → MMD 用の VMD

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamak493/nlf/blob/claude/kind-mendel-tdveor/mp4_to_mannequin_ja.ipynb)

`mp4_to_mannequin_ja.ipynb` は、入力の mp4 を URL（既定: `https://made-by-free.com/garando.mp4`）からダウンロードし、始点秒数〜終点秒数を指定するだけで

1. NLF v0.3.2 の学習済みモデル（自動ダウンロード）でモーションを抽出し（`motion.npz`）、
2. モーションを MMD 用の VMD に変換し（変換は [`nlf2vmd`](nlf2vmd/README.md)、仕様は [`vmd.md`](vmd.md)。GPU 不要のコマンドライン `python -m nlf2vmd` でも実行できます）、
3. 動画の音声から [Demucs](https://github.com/adefossez/demucs) でボーカルを取り出し、[Allosaurus](https://github.com/xinjli/allosaurus) で母音を認識して、口のモーフ（「あ」「い」「う」「え」「お」「ん」）のキーを作り、
4. 手首のまわりを切り出して [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) で指の 21 点を求め、手の形（グー・チョキ・パー・指 1 本だけ など 9 種）を判定して指ボーンのキーを作り、
5. VMD を **フル**（`motion_full.vmd`。既定）/ **フル [移動なし]** / **上半身のみ** / **表情のみ** から選んで書き出す

という一連の処理を行う日本語ノートブックです。出力は VMD だけです。SMPL の体モデルは NLF の TorchScript に入っているので、
SMPL 公式ファイルが無くても動作します。

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
