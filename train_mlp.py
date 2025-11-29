import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self, in_dim, h1, h2, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError('label column missing')
    X = df[[c for c in df.columns if c not in ['label', 'image_path']]].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    return X, y, le


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train(csv, output_dir, epochs, batch_size, lr, gpu, test_size, seed, h1, h2, dropout):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

    X, y, le = load_data(csv)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    in_dim = X_train.shape[1]
    out_dim = len(le.classes_)
    model = MLP(in_dim, h1, h2, out_dim, dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64)))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    metrics = []
    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc_sum = 0.0
        tr_cnt = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_acc_sum += accuracy(logits.detach(), yb) * xb.size(0)
            tr_cnt += xb.size(0)
        tr_loss /= tr_cnt
        tr_acc = tr_acc_sum / tr_cnt

        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        val_cnt = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_acc_sum += accuracy(logits, yb) * xb.size(0)
                val_cnt += xb.size(0)
        val_loss /= val_cnt
        val_acc = val_acc_sum / val_cnt
        metrics.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': val_loss, 'val_acc': val_acc})
        print(f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'in_dim': in_dim, 'h1': h1, 'h2': h2, 'out_dim': out_dim, 'dropout': dropout}, output_dir / 'mlp_state.pth')
    joblib.dump(scaler, output_dir / 'scaler.pkl')
    with open(output_dir / 'label_map.json', 'w', encoding='utf-8') as f:
        json.dump({'classes': le.classes_.tolist()}, f)
    pd.DataFrame(metrics).to_csv(output_dir / 'metrics_mlp.csv', index=False)
    print(f"saved_model={output_dir / 'mlp_state.pth'}")
    print(f"saved_scaler={output_dir / 'scaler.pkl'}")
    print(f"saved_label_map={output_dir / 'label_map.json'}")
    print(f"saved_metrics={output_dir / 'metrics_mlp.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=Path.cwd() / 'landmarks.csv')
    p.add_argument('--output-dir', type=Path, default=Path.cwd() / 'models')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hidden1', type=int, default=128)
    p.add_argument('--hidden2', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    args = p.parse_args()
    train(args.csv, args.output_dir, args.epochs, args.batch_size, args.lr, args.gpu, args.test_size, args.seed, args.hidden1, args.hidden2, args.dropout)


if __name__ == '__main__':
    main()

