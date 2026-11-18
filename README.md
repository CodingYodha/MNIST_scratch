# MNIST_scratch

A neural network for MNIST digit classification written from scratch, with no PyTorch, TensorFlow or Keras layers. Every part is hand-written: dense layers, activations, the loss, backpropagation, the optimizer and dropout.

The maths runs on **CuPy** (NumPy-style arrays on an NVIDIA GPU) so training is fast.

## How it works

All of this is in [`nn.ipynb`](nn.ipynb):

- **Dense layers** with a hand-written forward and backward pass
- **ReLU** activation
- **Softmax + categorical cross-entropy**, combined into one step for a simpler backward pass
- **Adam** optimizer with learning-rate decay
- **Dropout** (rate 0.2)
- **Evaluation:** confusion matrix, per-class accuracy, learning curve, and a view of misclassified digits

Keras is only used to download the MNIST dataset.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **98.17%** |
| Training accuracy (last epochs) | about 99.6% |

The learning curve and confusion matrix plots are in the notebook.

## How to run

You need an NVIDIA GPU with CUDA for CuPy.

```bash
pip install cupy-cuda12x numpy pandas matplotlib keras
jupyter notebook nn.ipynb
```

Pick the `cupy-cuda1xx` package that matches your CUDA version.
