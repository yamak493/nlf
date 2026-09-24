# nlf2vmd — NLF のモーションを MMD 用 VMD に変換する

NLF（[mp4_to_mannequin_ja.ipynb](../mp4_to_mannequin_ja.ipynb)）で抽出した SMPL のモーションを、
MikuMikuDance / MikuMikuMoving 用の VMD に変換します。仕様は [vmd.md](../vmd.md) です。

* 脚は**足ＩＫの位置キー**、体の移動は**センター（水平）／グルーブ（上下）**、上半身と腕は**回転キー**で出力します
* 接地中の足はロックし（足滑り 0）、床に埋まらないようにし、脚が伸び切らない高さまでセンターを下げます
* 処理前後の評価指標（JSON）と、判定の妥当性を確認するグラフ（PNG）を毎回出力します
* GPU も SMPL の公式ファイルも不要です（体モデルは NLF の TorchScript から書き出したものを使います）

## 使い方

### ノートブックから

`mp4_to_mannequin_ja.ipynb` のセル 14 を実行すると `nlf_mannequin/motion.vmd` ができます。
同じフォルダに、変換の入力 `motion_for_vmd.npz` と体モデル `smpl_body_model.npz` も保存されます。

### コマンドラインから

必要なもの: Python 3.9 以上、`numpy` `scipy` `pyyaml` `matplotlib`（グラフを出す場合）。

```bash
# ノートブックが保存した入力と体モデルから変換（体モデルは同じフォルダにあれば自動で使われます）
python -m nlf2vmd motion_for_vmd.npz -o motion.vmd --pmx モデル.pmx

# 設定を 1 項目ずつ変える
python -m nlf2vmd motion_for_vmd.npz --pmx モデル.pmx --set center.mode=B --set scale.depth_scale=0.5

# 設定ファイルで変える（default_config.yaml をコピーして、変えたい項目だけ残す）
python -m nlf2vmd --dump-config my_config.yaml
python -m nlf2vmd motion_for_vmd.npz --pmx モデル.pmx --config my_config.yaml

# 入力なしで動作確認（合成の歩行データと簡易体モデル）
python -m nlf2vmd --demo -o demo.vmd
```

### Python から

```python
from nlf2vmd import convert

result = convert('motion_for_vmd.npz', 'motion.vmd', pmx='モデル.pmx',
                 overrides=['center.mode=B'])
print(result.metrics['foot_slide_cm_per_frame'])   # {'before': ..., 'after': ...}
```

`convert` の入力は npz のパスか、同じキーを持つ dict です。

| キー | 形 | 内容 |
|---|---|---|
| `pose` | (T, 24, 3) | SMPL の関節回転（回転ベクトル）。(T, 24, 4) のクォータニオン、(T, 24, 3, 3) の回転行列も可。先頭に人物の次元があれば `input.person_index` の 1 人を使う |
| `betas` | (T, 10) か (10,) | 体型。フレーム毎なら中央値で 1 つに固定 |
| `trans` | (T, 3) | SMPL の移動量 [m] |
| `fps` | スカラー | |
| `coord_system` | 任意 | `camera`（NLF の出力: Y 下向き・Z 奥向き）か `yup`。無ければ `input.coords` の設定（既定 camera） |
| `valid` / `joints3d` | 任意 | 検出できたフレーム / 関節位置 [mm]（体モデルの自己検証に使う） |

体モデルは `--body-model`、入力 npz 内の `bm_*` キー、入力と同じフォルダの `smpl_body_model.npz`、
smplfitter（SMPL 公式ファイル）の順に探します。

## 出力

| MMD ボーン | キー | ソース |
|---|---|---|
| センター | 位置（X, Z） | 骨盤の水平位置（ステージ8） |
| グルーブ | 位置（Y） | 骨盤の高さ（ステージ8）。グルーブが無いモデルはセンターの Y |
| 下半身 / 上半身 / 上半身2 | 回転 | 骨盤(0) / 背骨(3, 6) / 背骨(9)。上半身2 が無いモデルは上半身に合成 |
| 首 / 頭 | 回転 | 首(12) / 頭(15) |
| 左右の肩・腕・ひじ・手首 | 回転 | 鎖骨(13, 14)・肩(16, 17)・肘(18, 19)・手首(20, 21) |
| 左足ＩＫ・右足ＩＫ | 位置＋回転 | 足首の位置と回転（ステージ7） |

足・ひざ・足首・つま先ＩＫ・捩りボーンにはキーを打ちません。キーは全フレームに打ちます（`vmd.thin_keys` で間引き可）。
PMX を指定しないときは、標準的な体格（身長 20 単位前後・A ポーズ）のボーン寸法で変換します。
**実際に使うモデルの PMX を指定するのがおすすめです**（脚長とボーンの向きをモデルから読みます）。

診断出力（既定: `<VMD 名>_diag/`）:

* `diagnostics.json` — 評価指標（処理前・処理後）、接地区間、床・スケール・警告などの情報、使った設定
* `contact.png` — 左右の足の高さ・水平速度と接地判定の帯
* `center.png` — センターの X・Y・Z の処理前後
* `reach_clamp.png` — 届く高さへのクランプの補正量
* `foot_ik_topview.png` — 足ＩＫの水平軌跡の上面図（接地点が点になっているか）

| 指標 | 定義 | 目標 |
|---|---|---|
| 足滑り | 接地区間内の足ＩＫの水平移動量の平均（cm/フレーム） | 0 |
| 埋まり | 足ＩＫの差分Yが0未満のフレーム数 | 0 |
| センターの震え | センター・グルーブ位置の加速度の絶対値の平均（軸別、cm/フレーム²） | 処理前より大幅に減少 |
| 姿勢の震え | 各関節の角加速度の平均（グループ別、deg/s²） | 処理前より減少 |
| 脚の伸び切り | 股関節から足ＩＫまでの距離が脚長の98%を超えるフレーム数 | 0 |
| 接地切り替え回数 | 足ごとの接地区間の数 | 目視との整合を確認 |

## 設定

既定値はすべて [default_config.yaml](default_config.yaml) にあり（説明付き）、設定ファイルか `--set` で上書きします。
長さのしきい値はメートルで書き、ステージ5以降は内部でスケール係数を掛けて使います。
存在しない項目を書くとエラーになります（タイプミス防止）。

MMD で見て問題があったときの調整の目安:

| 症状 | 調整する設定 |
|---|---|
| 体が後ろを向く・進行方向が逆 | `floor.flip_facing: true` |
| 足が滑る・接地のタイミングがずれる | `contact.png` を見て `contact.*` のしきい値 |
| 足が浮く・埋まる | `diagnostics.json` の `info.floor`。長尺で床の高さが変わるなら `floor.segment_mode: true` |
| センターが前後にふらつく | `scale.depth_scale`（0.5 など）、`center.axis_one_euro.z.min_cutoff` を小さく、`center.mode: B` |
| 膝が伸び切る・暴れる | `center.reach_ratio`（0.95 など） |
| 全身が震える／動きが鈍い | `jitter.one_euro.groups.*.min_cutoff` を小さく／大きく |

## 処理の構成

| ステージ | モジュール |
|---|---|
| 1. 読み込み・正規化 | `motion_io.py` |
| 2. 姿勢のジッター制御 | `jitter.py`（フィルタは `filters.py`、クォータニオンは `quat.py`） |
| 3. FK で関節位置を算出 | `body_model.py` |
| 4. 床面推定と定数オフセット | `floor.py` |
| 5. MMD スケールへ変換 | `pipeline.py`（`apply_scale`）、モデルの寸法は `pmx.py` / `skeleton.py` |
| 6. 接地判定 | `contact.py` |
| 7. 足ＩＫ生成 | `foot_ik.py` |
| 8. センター安定化 | `center.py` |
| 9. 上半身の回転リターゲット | `retarget.py` |
| 10. VMD 書き出しと診断出力 | `vmd.py`、`diagnostics.py` |

全体の順番は `pipeline.py` の `convert` にあります。順番は入れ替えないでください（vmd.md 参照）。

補足:

* One Euro フィルタはオフライン処理なので、順方向と逆方向の結果を平均して遅れを打ち消しています（`zero_phase`）。
  両端は点対称に延長してから掛けるので、一定速度で動いている区間の端にも加速度が出ません
* 床の傾き補正は、候補点が平面的に広がっているときだけ掛けます（まっすぐ歩くだけのクリップでは
  進行方向まわりの傾きが決まらないため。`floor.min_spread_m`）
* 届く高さへのクランプでは、ガウシアンのカーネルを移動最小値の窓の半分で打ち切っています。
  こうすると平滑化後の補正量がどのフレームでも必要量以上になり、クランプ後に脚が伸び切るフレームが残りません
* 補間パラメータ 64 バイトの配置は MMD Tools（blender_mmd_tools）の実装で確認したものです（`vmd.py` の先頭に記載）

## テスト

```bash
python -m pytest nlf2vmd/tests
```

合成データ（`synthetic.py`: 支持脚が床に固定された歩行、簡易体モデル）で、vmd.md の自動テスト
（フィルタ・接地判定・足ＩＫ・センター・VMD の読み戻し・座標変換）と、PMX の読み込み・設定の読み込みを確認します。
MMD での見た目の確認（足の滑り・埋まり・膝の暴れ・全身の震え）は手動で行ってください。
