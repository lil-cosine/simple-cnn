# CIFAR-10 CNN Classifier

A convolutional neural network trained on CIFAR-10 using PyTorch.
Three double-conv blocks (64→128→256 channels) with BatchNorm, dropout
regularisation, and a cosine-annealing LR schedule. Targets >80% test accuracy
in 30 epochs.

---

## Project structure

```
CNN/
├── main.py               # Model definition, training, evaluation
├── requirements.txt      # Python dependencies
├── best_model.pth        # Saved after training (git-ignored)
├── training_curves.png   # Loss & accuracy plots (generated)
└── predictions.png       # 4×4 prediction grid (generated)
```

The dataset lives outside this directory:

```
Datasets/
└── cifar-10-batches-py/  # Rename your CIFAR10/ folder to this
    ├── data_batch_1..5
    ├── test_batch
    └── batches.meta
```

> **Folder name matters.** `torchvision` looks for `cifar-10-batches-py/`
> inside the root you pass. Either rename the folder or create a symlink:
> ```bash
> ln -s ../Datasets/CIFAR10 ../Datasets/cifar-10-batches-py
> ```

---

## Setup

```bash
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Training

```bash
python main.py
```

Key hyperparameters (edit at the top of `__main__`):

| Parameter      | Default | Notes                          |
|----------------|---------|--------------------------------|
| `NUM_EPOCHS`   | 30      | Reduce to 5 for a smoke test   |
| `BATCH_SIZE`   | 128     | Lower if OOM on GPU            |
| `LEARNING_RATE`| 0.001   | AdamW, cosine-annealed to ~0   |
| `WEIGHT_DECAY` | 1e-4    | L2 regularisation              |

Training prints one line per epoch and saves `best_model.pth` whenever
validation accuracy improves.

Expected output:

```
Training on cuda
Epoch 001/30 | Train loss: 1.8643  acc: 31.3% | Val loss: 1.4344  acc: 48.2%  ✓ best
...
Epoch 030/30 | Train loss: 0.1865  acc: 93.8% | Val loss: 0.3102  acc: 89.9%  
```

---

## Outputs

After training completes the script writes three things:

- **`best_model.pth`** — `state_dict` of the highest val-accuracy checkpoint
- **`training_curves.png`** — side-by-side loss and accuracy plots
- **`predictions.png`** — 16 test images with predicted labels (green = correct)

---

## Architecture

```
Input 3×32×32
  → ConvBlock(3,   64)   → 64×16×16   # Conv-BN-ReLU ×2, MaxPool
  → ConvBlock(64, 128)   → 128×8×8
  → ConvBlock(128, 256)  → 256×4×4
  → Flatten → FC(4096,512) → Dropout(0.4)
           → FC(512, 256) → Dropout(0.4)
           → FC(256,  10)   logits
```

Weights initialised with Kaiming (conv) and Xavier (linear).

---

## Loading for inference

```python
import torch
from main import CIFAR10CNN

model = CIFAR10CNN()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()
```

---

## Results (30 epochs, RTX 500)

| Metric            | Value  |
|-------------------|--------|
| Best val accuracy | ~90% |
| Train time        | ~10 min |
| Parameters        | ~3.4 M  |
