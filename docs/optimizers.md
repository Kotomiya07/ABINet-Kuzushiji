# 利用可能なOptimizer一覧

このプロジェクトでは、PyTorchの`torch.optim`モジュールから動的にoptimizerを取得しているため、PyTorchで利用可能なすべてのoptimizerが使用できます。

## 主なOptimizer

### 1. **Adam** (推奨)
適応的モーメント推定。広く使用されており、多くの場合で良好な性能を示します。

```yaml
optimizer:
  type: Adam
  lr: 0.0001
  args: {
    betas: !!python/tuple [0.9, 0.999],  # デフォルト値
    eps: 1e-8,  # デフォルト値
    weight_decay: 0.0,  # デフォルト値
    amsgrad: False  # デフォルト値
  }
```

### 2. **AdamW**
Adamに重み減衰（Weight Decay）を組み合わせたoptimizer。正則化効果が高く、大規模モデルの学習に適しています。

```yaml
optimizer:
  type: AdamW
  lr: 0.0001
  args: {
    betas: !!python/tuple [0.9, 0.999],
    eps: 1e-8,
    weight_decay: 0.01,  # 重み減衰係数
    amsgrad: False
  }
```

### 3. **SGD** (Stochastic Gradient Descent)
確率的勾配降下法。シンプルで安定していますが、学習率の調整が必要です。

```yaml
optimizer:
  type: SGD
  lr: 0.01
  args: {
    momentum: 0.9,  # モーメンタム係数（推奨: 0.9）
    weight_decay: 0.0001,
    nesterov: False  # Nesterov加速勾配を使用するか
  }
```

### 4. **Adadelta**
学習率の減衰を防ぎ、より安定した学習を実現します。学習率の設定が不要です。

```yaml
optimizer:
  type: Adadelta
  lr: 1.0  # Adadeltaでは通常1.0を使用
  args: {
    rho: 0.9,  # デフォルト値
    eps: 1e-6,  # デフォルト値
    weight_decay: 0.0  # デフォルト値
  }
```

### 5. **RMSprop**
勾配の二乗平均を利用して学習率を調整し、勾配の振動を抑えます。

```yaml
optimizer:
  type: RMSprop
  lr: 0.01
  args: {
    alpha: 0.99,  # デフォルト値
    eps: 1e-8,  # デフォルト値
    weight_decay: 0.0,  # デフォルト値
    momentum: 0.0,  # デフォルト値
    centered: False  # デフォルト値
  }
```

### 6. **Adagrad**
パラメータごとに適応的な学習率を適用します。まれに更新されるパラメータの学習を促進します。

```yaml
optimizer:
  type: Adagrad
  lr: 0.01
  args: {
    lr_decay: 0.0,  # デフォルト値
    weight_decay: 0.0,  # デフォルト値
    eps: 1e-10,  # デフォルト値
    initial_accumulator_value: 0.0  # デフォルト値
  }
```

### 7. **Adamax**
Adamの変種で、無限ノルムベースの適応的学習率を使用します。

```yaml
optimizer:
  type: Adamax
  lr: 0.002
  args: {
    betas: !!python/tuple [0.9, 0.999],
    eps: 1e-8,
    weight_decay: 0.0
  }
```

### 8. **ASGD** (Averaged Stochastic Gradient Descent)
確率的勾配降下法の平均化版です。

```yaml
optimizer:
  type: ASGD
  lr: 0.01
  args: {
    lambd: 0.0001,  # デフォルト値
    alpha: 0.75,  # デフォルト値
    t0: 1000000.0,  # デフォルト値
    weight_decay: 0.0
  }
```

### 9. **Rprop**
Resilient backpropagation。符号ベースの適応的学習率を使用します。

```yaml
optimizer:
  type: Rprop
  lr: 0.01
  args: {
    etas: !!python/tuple [0.5, 1.2],  # デフォルト値
    step_sizes: !!python/tuple [1e-06, 50.0]  # デフォルト値
  }
```

### 10. **RAdam** (Rectified Adam)
Adamの修正版で、学習の初期段階での分散を修正します。

```yaml
optimizer:
  type: RAdam
  lr: 0.001
  args: {
    betas: !!python/tuple [0.9, 0.999],
    eps: 1e-8,
    weight_decay: 0.0
  }
```

### 11. **RAdamScheduleFree** (Schedule-Free RAdam)
学習率スケジューリングが不要なRAdamの変種。学習率の調整が不要で、安定かつ高速な収束を実現します。

```yaml
optimizer:
  type: RAdamScheduleFree
  lr: 0.001  # より大きな学習率が推奨される場合があります
  args: {
    betas: !!python/tuple [0.9, 0.999],  # デフォルト値。長期間のトレーニングでは[0.95, 0.999]や[0.98, 0.999]も検討
    eps: 1e-8,  # デフォルト値
    weight_decay: 0.0,  # デフォルト値
    r: 0.0,  # デフォルト値
    weight_lr_power: 2.0,  # デフォルト値
    foreach: True,  # デフォルト値
    silent_sgd_phase: True  # デフォルト値
  }
  scheduler: {
    # 注意: RAdamScheduleFreeでは学習率スケジューラは無視されます
    periods: [70, 10],
    gamma: 0.1,
  }
```

**特徴:**
- 学習率スケジューラが不要
- より大きな学習率が効果的
- トレーニングと評価時に`optimizer.train()`と`optimizer.eval()`を呼び出す必要がある（Lightningが自動的に処理）

### 12. **AdamWScheduleFree** (Schedule-Free AdamW)
学習率スケジューリングが不要なAdamWの変種。

```yaml
optimizer:
  type: AdamWScheduleFree
  lr: 0.001
  args: {
    betas: !!python/tuple [0.9, 0.999],
    weight_decay: 0.01,
    eps: 1e-8,
    r: 0.0,
    weight_lr_power: 2.0,
    foreach: True,
    silent_sgd_phase: True
  }
  scheduler: {
    # 注意: AdamWScheduleFreeでは学習率スケジューラは無視されます
    periods: [70, 10],
    gamma: 0.1,
  }
```

## 推奨設定

### 言語モデル事前学習
```yaml
optimizer:
  type: Adam  # または AdamW
  lr: 0.0001
  args: {
    betas: !!python/tuple [0.9, 0.999]
  }
```

### Visionモデル事前学習
```yaml
optimizer:
  type: Adadelta  # または Adam
  lr: 1.0  # Adadeltaの場合
  args: {}
```

### エンドツーエンド学習
```yaml
optimizer:
  type: Adam
  lr: 0.0001
  args: {
    betas: !!python/tuple [0.9, 0.999]
  }
```

## 注意事項

1. **学習率**: optimizerによって推奨される学習率が異なります
   - Adam/AdamW: 0.0001 ~ 0.001
   - SGD: 0.01 ~ 0.1
   - Adadelta: 1.0（学習率の設定が不要）
   - RAdamScheduleFree/AdamWScheduleFree: 0.001 ~ 0.01（より大きな学習率が推奨）

2. **重み減衰**: `optimizer.args.weight_decay`で設定できますが、`optimizer.wd`も設定されている場合は注意が必要です

3. **勾配クリッピング**: `optimizer.clip_grad`で設定可能です（デフォルト: 20）

4. **スケジューラ**: 
   - 標準のoptimizer: `optimizer.scheduler`で学習率スケジューラを設定できます
   - **RAdamScheduleFree/AdamWScheduleFree**: 学習率スケジューラは不要で、設定されても無視されます

5. **Schedule-Free Optimizer**: 
   - `schedulefree`ライブラリが必要です（既に依存関係に含まれています）
   - トレーニングと評価時に自動的に`optimizer.train()`と`optimizer.eval()`が呼び出されます
   - BatchNormが含まれるモデルでは、評価前に追加の処理が必要な場合があります

## 設定例

完全な設定例は`configs/`ディレクトリ内の各設定ファイルを参照してください。

