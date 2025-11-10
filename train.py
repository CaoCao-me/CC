import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

#这是两个损失函数  ICP正则 原型损失
from dpl_utils import NormIncreaseLoss, PrototypeLoss
from load_data import load_bcic
from model import SST_DPN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Lists to store accuracy for each phase
first_phase_train_acc = []
first_phase_val_acc = []
second_phase_train_acc = []

# Preprocessing settings for dataset
preprocessing_2a = {
    "sfreq": 250,  # Sampling frequency
    "low_cut": None,  # Low-frequency cutoff
    "high_cut": None,  # High-frequency cutoff
    "start": 0,  # Start time
    "stop": 0,  # Stop time
    "z_scale": False,  # Whether to apply z-scaling
}

# Load EEG data
subject = 5
X, y, X_test, y_test = load_bcic("2a", subject, preprocessing_2a)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Split data into training and validation sets
from sklearn.model_selection import train_test_split

X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors and move to the device
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long).to(device)
x_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)

# Create datasets and data loaders
train_dataset = TensorDataset(X, y)
val_dataset = TensorDataset(x_val, y_val)

# Define the model, loss function, and optimizers
sst_dpn = SST_DPN(
    chans=X.shape[1],  # Number of channels
    samples=X.shape[2],  # Number of samples
    num_classes=4,  # Number of classes
    F1=9,  # Number of filters in the first layer
    F2=48,  # Number of filters in the second layer
    time_kernel1=75,  # Time kernel size
    pool_kernels=[50, 100, 200],  # Pooling kernel sizes
).to(device)

model = sst_dpn.to(device)  # Move the model to the device

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.01
)  # Optimizer

# Optimizers for ISP (Inter-Subject Prototype) and ICP (Intra-Subject Prototype)
optimizer4isp = torch.optim.Adam([{"params": model.isp, "lr": 0.001}])
optimizer4icp = torch.optim.Adam([{"params": model.icp, "lr": 0.001}])


# Define functions to save and load model checkpoints
def save_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, filepath):
    """
    Save the model and optimizer states to a file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer4isp_state_dict": optimizer4isp.state_dict(),
        "optimizer4icp_state_dict": optimizer4icp.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, filepath):
    """
    Load the model and optimizer states from a file.
    """
    checkpoint = torch.load(
        filepath,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer4isp.load_state_dict(checkpoint["optimizer4isp_state_dict"])
    optimizer4icp.load_state_dict(checkpoint["optimizer4icp_state_dict"])


# Define the loss functions for ICP and Prototype Loss
loss_icp = NormIncreaseLoss()  # From dpl_utils.py
loss_pl = PrototypeLoss()  # From dpl_utils.py


# Define the validation function
def validate(model, val_loader):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == y_batch).sum().item()
            total_val += y_batch.size(0)
            loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            total_loss = loss + 0.001 * pl_loss + 0.00001 * icp_loss

            val_loss += loss.item()  # only cls loss for monitor
    first_phase_val_acc.append(correct_val / total_val)
    return val_loss / len(val_loader)


# Training parameters
N1, Ne, N2 = (
    1000,
    200,
    300,
)  # Epochs for first phase, early stopping patience, and second phase
patience = Ne

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# First training phase
best_loss = float("inf")
no_improve_epochs = 0

for epoch in range(N1):
    model.train()
    correct_train = 0
    total_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        optimizer4isp.zero_grad()
        optimizer4icp.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        icp_loss = loss_icp(model.icp)
        feature = model.get_features()
        proxy = model.icp
        pl_loss = loss_pl(feature, proxy, y_batch)
        total_loss = loss + 0.001 * pl_loss + 0.00001 * icp_loss
        total_loss.backward()
        optimizer.step()
        optimizer4isp.step()
        optimizer4icp.step()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)
    first_phase_train_acc.append(correct_train / total_train)
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch+1}/{N1}, Validation Loss: {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve_epochs = 0
        save_checkpoint(
            model, optimizer, optimizer4isp, optimizer4icp, "best_model.pth"
        )
    else:
        no_improve_epochs += 1
    if no_improve_epochs >= patience:
        print("Early stopping triggered")
        break

# Load the best model
load_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, "best_model.pth")

# Combine training and validation datasets for the second phase
train_dataset_full = TensorDataset(torch.cat([X, x_val]), torch.cat([y, y_val]))
train_loader_full = DataLoader(train_dataset_full, batch_size=32, shuffle=True)

# Second training phase
for epoch in range(N2):
    correct_train = 0
    total_train = 0
    model.train()
    for X_batch, y_batch in train_loader_full:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        optimizer4isp.zero_grad()
        optimizer4icp.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        icp_loss = loss_icp(model.icp)
        feature = model.get_features()
        proxy = model.icp
        pl_loss = loss_pl(feature, proxy, y_batch)
        total_loss = loss + 0.001 * pl_loss + 0.00001 * icp_loss
        total_loss.backward()
        optimizer.step()
        optimizer4isp.step()
        optimizer4icp.step()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == y_batch).sum().item()
        total_train += y_batch.size(0)
    second_phase_train_acc.append(correct_train / total_train)
    print(f"Epoch {epoch+1}/{N2}, Training Loss: {loss.item():.4f}")


# Test the model on the test dataset
def test_model(model, X_test, y_test):
    """
    Evaluate the model on the test dataset and print accuracy and Cohen's Kappa.
    """
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")


# Run the test evaluation
test_model(model, X_test, y_test)

"""
Since my original project was highly integrated, this training code has been simplified with the help of ChatGPT. 
I have tested it and confirmed that it can run directly, but I cannot guarantee its complete correctness.
"""



#----------------------------------------------我加的代码：
# python
import os
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int = 42):
    import random, torch, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_and_save(model, X_test, y_test, out_dir='results_subject5'):
    from sklearn.metrics import confusion_matrix, classification_report
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 推理
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test, dtype=torch.long).to(device)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            pred = out.argmax(1)
            preds.append(pred.cpu().numpy())
            labels.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    # 指标
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, output_dict=True)

    # 保存模型与指标
    torch.save(model.state_dict(), Path(out_dir) / 'final_model.pt')
    with open(Path(out_dir) / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({
            'subject': 5,  # 若改被试，修改此值或改为参数
            'accuracy': float(acc),
            'kappa': float(kappa),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'N1': int(N1), 'Ne': int(Ne), 'N2': int(N2)
        }, f, ensure_ascii=False, indent=2)

    # 保存预测与曲线
    np.save(Path(out_dir) / 'y_true.npy', labels)
    np.save(Path(out_dir) / 'y_pred.npy', preds)
    np.save(Path(out_dir) / 'first_phase_train_acc.npy', np.array(first_phase_train_acc))
    np.save(Path(out_dir) / 'first_phase_val_acc.npy', np.array(first_phase_val_acc))
    np.save(Path(out_dir) / 'second_phase_train_acc.npy', np.array(second_phase_train_acc))

    print(f"[保存完成] 目录: {out_dir}")
    print(f"Acc: {acc:.4f}  Kappa: {kappa:.4f}")
    print("Confusion Matrix:\n", cm)

# 建议：脚本开头尽早调用，保证复现
set_seed(42)

# 在已有 test_model(model, X_test, y_test) 调用之后再执行：
evaluate_and_save(model, X_test, y_test, out_dir='results_subject5')