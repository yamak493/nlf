# ライセンスと利用条件

このリポジトリと [`mp4_to_mannequin_ja.ipynb`](mp4_to_mannequin_ja.ipynb)（mp4 → MMD 用 VMD）が使っている
ソフトウェア・学習済みモデル・学習データのライセンスと、それを踏まえた利用条件をまとめたものです（2026-09-28 時点）。

> 法的な助言ではありません。各ライセンスの原文（下の「参照」）が優先します。

## 利用条件（まとめ）

このノートブックとそれで作ったモーションは、次の条件の範囲でだけ使えます。

1. **非商用に限る**。販売・有料配布・有料サービス、収益化した動画や配信（YouTube の収益化、ニコニコのクリエイター奨励プログラム、投げ銭、メンバー限定・有料支援者限定の公開など）、企業案件での使用はできません。
2. **ポルノ・軍事・監視の目的には使わない**（非商用でも不可。R-18 の作品も含みます）。
3. **虚偽・名誉毀損・誤解を招くコンテンツを作らない**（実在の人物がしていない動きをしたように見せる、特定の人物を中傷する、など）。
4. **作品を公開するときは NLF のクレジットを入れ、論文を引用する**（下の「クレジットの例」）。
5. 入力する**動画・音楽・PMX モデルの権利は利用者が確認する**（下の「素材の権利」）。
6. **SMPL の体モデルのデータを他人に渡さない**（下の「共有してはいけないファイル」）。

## NLF の作者の見解（isarandi/nlf#51）

NLF の学習済みモデルは README で「noncommercial research use」として公開されています。趣味の 3D アニメーション
（Blender・MMD）への利用について、上の 1・2・4 を条件に作者へ質問し、次の回答を得ています
（[isarandi/nlf#51](https://github.com/isarandi/nlf/issues/51)、2026-09-27）。

> The important thing is the noncommercial part. So it shouldn't be a sold product or service etc.
> "Research" has a broad meaning and can also include experimenting with tools for artistic purposes.
> So this is not a problem.

つまり、**非商用であれば、芸術目的でツールを試すこと（趣味の MMD 制作を含む）は「research」に含まれる**というのが
作者の見解です。重要なのは非商用であること（販売する製品やサービスにしないこと）です。
この回答は正式なライセンス文ではなく、Issue でのコメントです。

## 学習済みモデル・体モデル

ノートブックはどのモデルも同梱しておらず、実行時に各配布元からダウンロードします。
ダウンロードして使った時点で、各ライセンスに同意したことになります。

| 名前 | 取得元 | ライセンス | 主な制約 |
|---|---|---|---|
| NLF `nlf_l_multi_0.3.2.torchscript` | [GitHub Releases](https://github.com/isarandi/nlf/releases/tag/v0.3.2) | 非商用の研究用途（README の記載と #51 の回答） | 非商用に限る |
| SMPL / SMPL-X / MANO / FLAME（NLF の TorchScript に同梱） | 同上 | Max Planck の独自ライセンス | 非商用に限る・ポルノ／軍事／監視の禁止・虚偽／名誉毀損の禁止・再配布の禁止（下の「SMPL 系」） |
| YOLOv8x（人物検出。NLF の TorchScript に同梱） | 同上 | AGPL-3.0（Ultralytics） | 再配布するとき・ネットワーク越しのサービスにするときはソースを公開する義務 |
| Demucs `htdemucs`（ボーカルの分離） | Demucs が自動ダウンロード | MIT | 特になし |
| Allosaurus `uni2005`（音素の認識） | Allosaurus が自動ダウンロード | GPL-3.0（リポジトリのライセンス。重みに別の表記は無い） | 同梱・改変して配布するときは GPL |
| MediaPipe Hands `hand_landmarker.task` | Google Cloud Storage | Apache-2.0 | 特になし |

### SMPL 系

SMPL・SMPL-X・MANO・FLAME のライセンスは、どれも次の目的だけに使うことを認めています。

> non-commercial scientific research, non-commercial education, or non-commercial artistic projects

非商用の MMD 作品は「non-commercial artistic projects」に当たります。一方で、次のことは禁止されています。

* 商用利用（「production of other artefacts for commercial purposes」を含む。つまり、作ったモーションを使った作品の収益化も不可）
* ポルノ目的・ポルノの生成（「whether commercial or not」）
* 軍事・監視の目的（SMPL-X・MANO・FLAME）
* 虚偽・名誉毀損・誤解を招くコンテンツの作成（SMPL-X・MANO・FLAME）
* 第三者への再配布（アーカイブ用の複製 1 部を除く）
* 商用・ポルノ・軍事・監視・名誉毀損の目的で使う手法（ニューラルネットワークなど）の学習

ノートブックは SMPL の公式ファイルを別にダウンロードしませんが、NLF の TorchScript に入っている SMPL を使うので、
**SMPL のライセンス（[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/modellicense.html)）は同じように適用されます**。

### 学習データ

NLF は約 40 のデータセットで学習されています（Human3.6M・MPI-INF-3DHP・3DPW・AGORA・BEDLAM・SURREAL・EgoBody・
BEHAVE・RICH・HuMMan・GeneBody・DNA-Rendering・Hi4D・SAIL-VOS・JTA・DensePose-COCO など）。
バックボーンの EfficientNetV2 は ImageNet で事前学習されています。これらの多くは学術・非商用に限った規約で公開されており、
NLF のモデルが非商用に限られている理由のひとつです。

Demucs `htdemucs` は MUSDB18-HQ（非商用の研究用）と Meta 社内の楽曲 800 曲で、MediaPipe Hands は約 3 万枚の実画像と
合成画像で学習されています。どちらもモデル自体のライセンス（MIT・Apache-2.0）の範囲で使えます。

## ソフトウェア

| 名前 | ライセンス | 備考 |
|---|---|---|
| NLF のコード（このリポジトリの元） | MIT（© 2024 István Sárándi、[`LICENSE`](LICENSE)） | 配布するときは著作権表示を残す |
| `nlf2vmd`・`mp4_to_mannequin_ja.ipynb` | このリポジトリの追加分 | 使うモデルの制約により、実際の利用は上の「利用条件」の範囲に限られる |
| PyTorch・torchvision | BSD-3-Clause | |
| NumPy・SciPy・imageio・IPython | BSD 系 | |
| matplotlib | Matplotlib License（PSF 系） | |
| tqdm | MPL-2.0 と MIT | |
| ai-edge-litert（LiteRT） | Apache-2.0 | MediaPipe のモデルを動かすランタイム |
| demucs | MIT | |
| allosaurus | GPL-3.0 | ノートブックは pip で入れて呼び出すだけで、`nlf2vmd` は出力のテキストを読むだけなので、このリポジトリのコードには及ばない |
| imageio-ffmpeg | BSD-2-Clause | 同梱の ffmpeg バイナリは **GPL-3.0**（`--enable-gpl --enable-version3 --enable-libx264`）。使うだけなら問題なく、ffmpeg を再配布するときに GPL の義務がある |

## 共有してはいけないファイル

ノートブックが書き出すファイルのうち、次のものは他人に渡さないでください。

| ファイル | 理由 |
|---|---|
| `smpl_body_model.npz` | SMPL の本体（テンプレートの頂点・体型のブレンドシェイプ・スキニングの重み）。SMPL のライセンスで再配布が禁止されている |
| `motion.npz`（`vertices3d` を含む） | SMPL のメッシュの頂点。上と同じ扱いにする |
| `segment_audio.wav`・`vocals.wav` | 元の楽曲の音声（楽曲の著作権） |
| `nlf_l_multi_0.3.2.torchscript`・`hand_landmarker.task` | 各配布元から取得してもらう |

VMD（`motion_full.vmd` など）は MMD のボーンの回転・モーフのキーだけで、SMPL のデータは入っていません。
配布するときは、受け取った人にも上の「利用条件」（非商用・ポルノ／軍事／監視の禁止など）を守ってもらうよう明記してください。

## 素材の権利

ライセンスとは別に、MMD でいちばん問題になりやすいのは入力の素材です。

* **元動画**: 振付には著作権が認められることがあり、踊っている人には肖像権・パブリシティ権があります。
  自分で撮った動画か、踊り手・振付師の許可がある動画を使ってください（MMD では、トレース元の許可を取るのが慣習です）。
  他人を無断で撮影した動画の解析は、監視目的の禁止にも触れるおそれがあります。
* **音楽**: 口パクのために切り出した音声には楽曲の著作権があります。
* **PMX モデル**: モデルごとに利用規約（R-18・暴力表現・特定のモーションの禁止など）があるので、それに従ってください。

## クレジットの例

作品を公開するときは、説明欄などに次のように書いてください。

```
モーション: NLF (Neural Localizer Fields) で動画から推定
  István Sárándi and Gerard Pons-Moll, "Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation", NeurIPS 2024
  https://github.com/isarandi/nlf
体モデル: SMPL (Max Planck Institute for Intelligent Systems) https://smpl.is.tue.mpg.de/
```

論文などで使うときは、ノートブックの「参考」にある BibTeX（NLF と SMPL）を引用してください。

## 参照

* NLF: [README](https://github.com/isarandi/nlf)・[作者の回答 #51](https://github.com/isarandi/nlf/issues/51)・[論文](https://arxiv.org/abs/2407.07532)
* [SMPL-Model License](https://smpl.is.tue.mpg.de/modellicense.html)・[SMPL-Body License](https://smpl.is.tue.mpg.de/bodylicense.html)
* [SMPL-X License](https://smpl-x.is.tue.mpg.de/modellicense.html)・[MANO License](https://mano.is.tue.mpg.de/license.html)・[FLAME License](https://flame.is.tue.mpg.de/modellicense.html)
* [Ultralytics YOLOv8（AGPL-3.0）](https://github.com/ultralytics/ultralytics)
* [Demucs](https://github.com/adefossez/demucs)・[Allosaurus](https://github.com/xinjli/allosaurus)・[MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
* [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)・[FFmpeg のライセンス](https://ffmpeg.org/legal.html)
