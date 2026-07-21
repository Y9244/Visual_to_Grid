# グリッド細胞の等角等長性

[English README](README.md)

このリポジトリは、論文
[On Conformal Isometry of Grid Cells: Learning Distance-Preserving Position Embedding](https://arxiv.org/abs/2405.16865)
の公式実装をもとにしています。

現在の実装は、六角格子状のグリッド細胞活動を安定して学習できた、次の条件に
一本化されています。

- 位置を連続的に表現するSIRENエンコーダと、非負な活動を出力するSoftplus
- `[cos(theta), sin(theta)]` から移動変換行列を生成するMLP
- サンプルごとにランダムな移動方向を1つ使用
- isometry Lossで使う移動距離を `dr = 5` に固定
- 移動前後の活動の距離二乗に対するisometry Loss
- Lossを使った活動ノルム制約

学習処理を読みやすくするため、以前の活動テーブル、通常のMLPエンコーダ、
角度bin、移動距離が変化するisometry Loss、複数方向を同時に評価する実験分岐は
削除されています。

## 環境構築

Python 3.12に対応しています。次のコマンドで依存関係をインストールします。

```bash
uv sync
```

## 学習

現在サポートしている設定は、次のコマンドで実行できます。

```bash
uv run python main.py \
  --config=configs/siren_scale10.py \
  --workdir=./logs
```

実行するたびに、`logs` 以下にタイムスタンプ付きの実行ディレクトリが作られます。
学習指標は `metrics.csv` に保存されます。PyTorchチェックポイントは500ステップ
ごとに `ckpt/checkpoint-step*.pth` へ保存され、設定とNumPyの乱数状態は同名の
JSONファイルへ保存されます。`.pth` ファイルは
`torch.load(..., weights_only=True)` で読み込めます。

## 可視化

学習終了後、`RUN_DIR` をタイムスタンプ付きの実行ディレクトリに置き換えて、
次のコマンドを実行します。

```bash
uv run python visualize_results.py RUN_DIR
```

特定のチェックポイントを可視化する場合は、`--step 5000` のように指定します。
受容野、方向に応じて生成された移動変換行列、Loss、活動数、および活動ノルムの
診断結果が `RUN_DIR/visualizations` 以下に保存されます。

六角格子と正方格子のgridnessは、次のコマンドで計算できます。

```bash
uv run python analyze_gridness.py RUN_DIR
```

gridnessスコアのCSVと空間自己相関の画像は、`RUN_DIR/gridness` 以下に
保存されます。
