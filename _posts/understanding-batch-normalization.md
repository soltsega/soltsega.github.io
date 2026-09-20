---
layout: post
title: "Understanding Batch Normalization: From Intuition to Implementation"
date: 2026-09-20
categories: [deep-learning, optimization]
tags: [batch-normalization, neural-networks, training-stability, regularization]
excerpt: "Why batch normalization was invented, what it actually does mathematically, and how to implement and use it correctly."
---

<!-- MathJax script for rendering LaTeX equations -->
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Understanding Batch Normalization: From Intuition to Implementation

Batch normalization (BatchNorm) is one of the few ideas in deep learning that's both nearly universal in practice and still genuinely debated in theory. It was introduced in 2015 to make deep networks train faster and more reliably, and it's now a default layer in most CNN architectures. This post covers it in two passes: intuition first, then the full math, gradients, and implementation.

---

## Part 1: Intuition

### The problem BatchNorm solves

Training a deep neural network means stacking many layers, each transforming its input before passing it to the next. As training proceeds, the parameters of every layer are updated simultaneously by gradient descent. This creates a subtle problem: the *distribution* of inputs that a given layer receives keeps shifting, because every layer below it is also changing.

The original BatchNorm paper called this **internal covariate shift** — each layer has to constantly re-adapt to a moving target, because the statistics (mean, variance) of its inputs drift as earlier layers' weights update. This slows down training: you're forced to use a small learning rate and careful initialization just to keep things stable, and even then, deep networks can be painfully slow to converge, or fail to converge at all.

### The core idea

BatchNorm's fix is direct: at each layer, explicitly force the inputs (technically, the pre-activation values) to have zero mean and unit variance, computed **per mini-batch** during training. Then, to make sure this normalization doesn't strip away useful information, add back two learnable parameters per feature — a scale and a shift — so the network can undo the normalization if that's actually what's best for a given feature.

Concretely, for a mini-batch of activations, BatchNorm:

1. Computes the mean and variance of each feature across the batch.
2. Normalizes each feature to zero mean, unit variance using those statistics.
3. Applies a learned scale ($\gamma$) and shift ($\beta$) so the layer retains full representational flexibility.

### Why this helps, intuitively

- **Smoother optimization landscape.** Later research (Santurkar et al., 2018) showed the classical "reduces internal covariate shift" explanation is incomplete — the bigger effect seems to be that BatchNorm makes the loss surface smoother and gradients more predictable, letting you safely use much higher learning rates.
- **Less sensitivity to initialization.** Because activations are re-centered and re-scaled at every layer, poor weight initialization is less likely to send activations into a regime where gradients vanish or explode.
- **A mild regularization effect.** Because the batch statistics are noisy estimates (computed from a random mini-batch rather than the whole dataset), BatchNorm injects a small amount of noise into training, which acts a bit like a regularizer — this is why networks with BatchNorm sometimes need less dropout.
- **Enables much deeper networks.** Before BatchNorm, training very deep networks reliably was hard. BatchNorm (along with residual connections) was one of the key ingredients that made 50+ layer networks practical.

### The catch: train vs. test behavior differs

At **training time**, BatchNorm uses the mean and variance of the *current mini-batch*. But at **test time**, you often want to run the model on a single example, or you want deterministic output — there's no "batch" to compute statistics from, and you don't want output to depend on which other examples happen to be in the batch. So BatchNorm keeps a running (exponential moving) average of the mean and variance seen during training, and uses *that* fixed estimate at inference time instead. This train/test asymmetry is the single most common source of BatchNorm-related bugs in practice (forgetting to switch a model into `eval()` mode, for instance).

If you only read Part 1, that's enough to use BatchNorm correctly day-to-day. Part 2 covers the exact math, the backward pass, and implementation details.

---

## Part 2: The Rigorous Version

### 2.1 The forward pass

For a mini-batch $\mathcal{B} = \{x_1, \dots, x_m\}$ of activations for a single feature (this is applied independently per feature/channel), BatchNorm computes:

**Batch mean:**
$$
\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

**Batch variance:**
$$
\sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2
$$

**Normalization:**
$$
\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}
$$

$\epsilon$ is a small constant (e.g. $10^{-5}$) added purely for numerical stability, to avoid dividing by zero when variance is tiny.

**Scale and shift:**
$$
y_i = \gamma \hat{x}_i + \beta
$$

$\gamma$ and $\beta$ are learned parameters, one pair per feature/channel, updated by gradient descent just like any other network weight. Note that if the optimizer decides normalization isn't helpful for a particular feature, it can learn $\gamma = \sqrt{\sigma_\mathcal{B}^2 + \epsilon}$ and $\beta = \mu_\mathcal{B}$ to exactly recover the original unnormalized values — so BatchNorm never strictly reduces the model's representational capacity.

### 2.2 Where it sits in the network

For a fully-connected layer, BatchNorm is applied per-feature across the batch dimension. For a convolutional layer, it's applied per-channel, with statistics computed across the batch *and* spatial dimensions (height, width) jointly — so every channel gets one $\mu$, one $\sigma^2$, one $\gamma$, one $\beta$, shared across all spatial locations.

The standard placement is: `Linear/Conv → BatchNorm → Activation`. That is, normalize the pre-activation values before applying the nonlinearity (e.g. ReLU). This ordering is what the original paper proposed, though some later architectures experiment with BatchNorm after the activation instead.

### 2.3 The backward pass

This is the part most tutorials skip, but it's important for understanding why BatchNorm affects gradients throughout the whole network, not just locally. Given the upstream gradient $\partial L/\partial y_i$, we need $\partial L/\partial x_i$, plus gradients for $\gamma$ and $\beta$.

**Gradients for the learned parameters:**
$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^m \frac{\partial L}{\partial y_i} \cdot \hat{x}_i, \qquad
\frac{\partial L}{\partial \beta} = \sum_{i=1}^m \frac{\partial L}{\partial y_i}
$$

**Gradient with respect to the normalized value:**
$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma
$$

**Gradient with respect to the batch variance:**
$$
\frac{\partial L}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \left(-\frac{1}{2}\right)(\sigma_\mathcal{B}^2 + \epsilon)^{-3/2}
$$

**Gradient with respect to the batch mean:**
$$
\frac{\partial L}{\partial \mu_\mathcal{B}} = \left(\sum_{i=1}^m \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}\right) + \frac{\partial L}{\partial \sigma_\mathcal{B}^2} \cdot \frac{\sum_{i=1}^m -2(x_i - \mu_\mathcal{B})}{m}
$$

**Gradient with respect to each input:**
$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial L}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}
$$

The key takeaway: because $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$ are themselves functions of *every* example in the batch, the gradient for a single example $x_i$ depends on all the other examples in the batch too. This is what gives BatchNorm its batch-dependent, slightly stochastic character, and it's also why very small batch sizes hurt BatchNorm's effectiveness — the statistics become noisy, high-variance estimates.

### 2.4 Running statistics for inference

During training, alongside the per-batch $\mu_\mathcal{B}$ and $\sigma_\mathcal{B}^2$, the layer maintains running estimates updated with an exponential moving average:

$$
\mu_{\text{running}} \leftarrow (1 - \alpha)\, \mu_{\text{running}} + \alpha\, \mu_\mathcal{B}
$$
$$
\sigma^2_{\text{running}} \leftarrow (1 - \alpha)\, \sigma^2_{\text{running}} + \alpha\, \sigma^2_\mathcal{B}
$$

where $\alpha$ (often called `momentum` in framework APIs — note this is unrelated to optimizer momentum) is typically around 0.1. At inference time, the layer uses $\mu_{\text{running}}$ and $\sigma^2_{\text{running}}$ instead of computing fresh batch statistics, since a single test example (or a differently-sized batch) shouldn't produce different outputs depending on what else happens to be in that batch.

### 2.5 Implementation

#### Using PyTorch (the practical approach)

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)  # one gamma/beta per channel
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

model = nn.Sequential(
    ConvBlock(3, 64),
    ConvBlock(64, 128),
)

model.train()   # uses batch statistics, updates running estimates
# ... training loop ...

model.eval()    # uses running statistics, deterministic output
with torch.no_grad():
    output = model(torch.randn(1, 3, 32, 32))
```

Two details worth calling out:
- `bias=False` on the convolution: since BatchNorm re-centers with its own learned $\beta$, a separate conv bias term is redundant and is usually dropped.
- The `model.train()` / `model.eval()` switch is what controls whether BatchNorm uses batch statistics or running statistics — forgetting this before evaluation is one of the most common PyTorch bugs.

#### Using TensorFlow/Keras — image recognition on MNIST

A simple, readable example that puts BatchNorm between every dense layer — exactly the pattern that shows its effect most clearly:

```python
import tensorflow as tf
from tensorflow import keras

# ── 1. Load and split data ────────────────────────────────────────────────────
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reserve the first 5 000 samples as a validation set; normalise to [0, 1]
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000],          y_train_full[5000:]
X_test = X_test / 255.0

# ── 2. Build the model ────────────────────────────────────────────────────────
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),          # normalise raw pixel inputs
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),          # normalise before next layer
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),          # normalise before output
    keras.layers.Dense(10, activation="softmax"),
])

model.summary()

# ── 3. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer="sgd",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ── 4. Train ──────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_valid, y_valid),
)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
# model.evaluate() automatically uses running statistics — no extra step needed
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")   # typically ~97–98 %

# ── 6. Predict a single image ─────────────────────────────────────────────────
import numpy as np
sample = X_test[0:1]                           # keep batch dimension: (1, 28, 28)
probs  = model.predict(sample, verbose=0)[0]   # inference mode: uses running stats
print(f"Predicted digit: {np.argmax(probs)}  "
      f"(confidence: {probs.max():.2%})")
```

A few things to notice here:

- **BatchNorm on the inputs.** The very first `BatchNormalization()` layer normalises the raw (already scaled) pixel values before the first `Dense`. This is especially useful when input features have different scales, though for MNIST — which is already uniform — the effect is mild.
- **BatchNorm between hidden layers.** Each `BatchNormalization()` re-centres and re-scales the pre-activation values of the *next* layer, stabilising the training signal as it propagates deeper.
- **No `bias=False` needed here.** Unlike `Conv2D`, Keras's `Dense` layers placed *before* a `BatchNormalization` could have `use_bias=False` (the BN $\beta$ subsumes it), but for this simple demo it's omitted for readability.
- **Inference is automatic.** `model.predict()` and `model.evaluate()` use the running mean/variance accumulated during training — no `model.eval()` call needed (unlike PyTorch).

#### From-scratch implementation (for understanding, not production)

```python
import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        # x has shape (batch_size, num_features)
        if training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            self.x_centered = x - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.eps)
            x_hat = self.x_centered * self.std_inv

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        self.x_hat = x_hat
        return self.gamma * x_hat + self.beta

    def backward(self, dout):
        m = dout.shape[0]

        dgamma = np.sum(dout * self.x_hat, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * self.x_centered * -0.5 * self.std_inv**3, axis=0)
        dmean = np.sum(dx_hat * -self.std_inv, axis=0) + dvar * np.mean(-2.0 * self.x_centered, axis=0)

        dx = (dx_hat * self.std_inv) + (dvar * 2.0 * self.x_centered / m) + (dmean / m)

        return dx, dgamma, dbeta
```

This mirrors the equations in Section 2.3 directly, term by term, so it doubles as a way to sanity-check the derivation.

### 2.6 Known issues and variants

- **Small batch sizes hurt it.** With very small batches (e.g. batch size 2–4, common in object detection or segmentation where images are large), the batch mean/variance are noisy, unreliable estimates, and BatchNorm can actively hurt training.
- **Sequence models don't play well with it.** In RNNs/Transformers, batches have variable-length sequences and statistics that shift across time steps, which BatchNorm handles poorly. This is why **LayerNorm** (which normalizes across the feature dimension for a single example, not across the batch) is the standard choice in Transformers instead.
- **Train/inference mismatch.** As covered above, batch stats vs. running stats can diverge if the running average hasn't converged well (e.g. training stopped too early, or momentum set poorly), leading to a training/inference performance gap.
- **Common alternatives**, each solving BatchNorm's batch-dependence in a different way:
  - **LayerNorm** — normalizes across features for each individual example; standard in Transformers.
  - **GroupNorm** — splits channels into groups and normalizes within each group, independent of batch size; useful for small-batch vision tasks.
  - **InstanceNorm** — normalizes each channel of each individual example separately; common in style-transfer models.

### 2.7 BatchNorm vs. alternatives at a glance

| | BatchNorm | LayerNorm | GroupNorm | InstanceNorm |
|---|---|---|---|---|
| Normalizes across | Batch (+ spatial, for conv) | Features, per example | Channel groups, per example | Spatial, per channel, per example |
| Batch-size sensitive | Yes | No | No | No |
| Typical use case | CNNs (large batch) | Transformers, RNNs | CNNs with small batches | Style transfer |
| Needs running stats for inference | Yes | No | No | No |

### 2.8 Notes I would like to share

- Place BatchNorm between the linear/conv layer and the activation function; drop the linear/conv layer's bias term.
- Always call `model.train()` / `model.eval()` correctly around training vs. evaluation/inference code.
- Avoid very small batch sizes with BatchNorm (below ~8 is often risky); switch to GroupNorm or LayerNorm if your batch size is constrained.
- For Transformers and sequence models, prefer LayerNorm over BatchNorm by default.
- If train and validation performance diverge oddly, check whether it's actually a train/eval mode bug before assuming it's overfitting.

---

## Further reading

- Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML.
- Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). *How Does Batch Normalization Help Optimization?* NeurIPS. (The paper that challenged the original "internal covariate shift" explanation.)
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv.
- Wu, Y. & He, K. (2018). *Group Normalization*. ECCV.

---

*Next up in this series: a look at LayerNorm and why it became the default in Transformer architectures instead of BatchNorm.*
