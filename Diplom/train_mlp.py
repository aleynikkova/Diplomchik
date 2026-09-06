import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

H5 = "pendulum_dataset.h5"
DTYPE = np.float32

class OneStepDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MLP(nn.Module):
    def __init__(self, in_dim=12, out_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_xy(h5_path=H5, dt=0.01):
    with h5py.File(h5_path, "r") as f:
        params = np.stack([
            f["params/m1"][:],
            f["params/m2"][:],
            f["params/L1"][:],
            f["params/L2"][:],
            f["params/b1"][:],
            f["params/b2"][:],
            f["params/theta1_0"][:],
            f["params/theta2_0"][:],
        ], axis=1).astype(DTYPE)  # (N, 8)

        #(N, T)
        th1 = f["trajectories/theta1"][:].astype(DTYPE)
        th2 = f["trajectories/theta2"][:].astype(DTYPE)
        dth1 = f["trajectories/dtheta1"][:].astype(DTYPE)
        dth2 = f["trajectories/dtheta2"][:].astype(DTYPE)

    N, T = th1.shape
   
    #(N, T-1, 4)
    Xs = np.stack([th1[:, :-1], th2[:, :-1], dth1[:, :-1], dth2[:, :-1]], axis=2)
    Xs_next = np.stack([th1[:, 1:], th2[:, 1:], dth1[:, 1:], dth2[:, 1:]], axis=2)

    #dY = Ys_next - Xs
    dstate = (Xs_next - Xs) / dt

    #(N, T-1, 8)
    Xp = np.repeat(params[:, None, :], T-1, axis=1)

    #(N*(T-1), dim)
    X = np.concatenate([Xp, Xs], axis=2).reshape(-1, 12)
    Y = dstate.reshape(-1, 4)

    return X, Y, dt

def main():
    X, Y, dt_train = load_xy(H5)
    print(f"Временной шаг в данных: dt = {dt_train} сек")
    print("X shape:", X.shape, "Y shape:", Y.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    Y_train_s = y_scaler.fit_transform(Y_train)
    X_val_s   = x_scaler.transform(X_val)
    Y_val_s   = y_scaler.transform(Y_val)

    joblib.dump(x_scaler, "x_scaler.pkl")
    joblib.dump(y_scaler, "y_scaler.pkl")

    train_ds = OneStepDataset(X_train_s, Y_train_s)
    val_ds   = OneStepDataset(X_val_s, Y_val_s)

    train_dl = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=0)
    
    #print(len(train_ds), train_ds[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, 31):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_ds)

        print(f"epoch {epoch:02d}: train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), "mlp_best.pt")

    print("Saved: mlp_best.pt, x_scaler.pkl, y_scaler.pkl")

if __name__ == "__main__":
    main()