import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    return total_loss / total, 100.0 * correct / total

def per_class_accuracy(model, loader, classes, device):
    model.eval()
    class_correct = [0] * len(classes)
    class_total   = [0] * len(classes)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(labels.size(0)):
                lbl = labels[i].item()
                class_correct[lbl] += (predicted[i] == lbl).item()
                class_total[lbl]   += 1

    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        acc = 100.0 * class_correct[i] / class_total[i]
        bar = '█' * int(acc // 5)
        print(f"  {cls:12s} {acc:5.1f}%  {bar}")

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', color='#6c7fff')
    axes[0].plot(history['val_loss'],   label='Val',   color='#f87171')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', color='#6c7fff')
    axes[1].plot(history['val_acc'],   label='Val',   color='#f87171')
    axes[1].set_title('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)

def show_predictions(model, loader, classes, device, n=16):
    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:n].to(device)

    with torch.no_grad():
        outputs = model(images_dev)
        _, preds = outputs.max(1)

    # Denormalise for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std  = torch.tensor([0.2470, 0.2435, 0.2616])
    imgs = images[:n] * std[:, None, None] + mean[:, None, None]
    imgs = imgs.permute(0, 2, 3, 1).numpy().clip(0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i])
        pred = preds[i].item()
        true = labels[i].item()
        color = 'green' if pred == true else 'red'
        ax.set_title(f"{classes[pred]}", color=color, fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Training on {device}")

    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 30
    BATCH_SIZE = 128

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std= (0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std= (0.2470, 0.2435, 0.2616)
    ),
    ])
    DATA_ROOT   = "../Datasets"
    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=True,
        transform=transform_train, download=False
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT, train=False,
        transform=transform_test, download=False
    )

    BATCH_SIZE = 128
    CLASSES = train_dataset.classes

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    CLASSES = train_dataset.classes

    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_acc     = 0.0
    history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = eval_epoch (model, test_loader,  criterion, device)
        scheduler.step()

        history['train_loss'].append(t_loss)
        history['val_loss'  ].append(v_loss)
        history['train_acc' ].append(t_acc)
        history['val_acc'   ].append(v_acc)

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train loss: {t_loss:.4f}  acc: {t_acc:.1f}% | "
            f"Val loss: {v_loss:.4f}  acc: {v_acc:.1f}%  "
            f"{'✓ best' if v_acc == best_acc else ''}")

    per_class_accuracy(model, test_loader, CLASSES, device)
    plot_history(history)
    show_predictions(model, test_loader, CLASSES, device)

