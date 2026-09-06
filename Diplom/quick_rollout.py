import h5py
import numpy as np
import torch
from torch import nn
import joblib
import matplotlib.pyplot as plt

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

def main(sample_id=0, steps=80, h5_path="pendulum_dataset.h5"):
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP().to(device)
    model.load_state_dict(torch.load("mlp_best.pt", map_location=device))
    model.eval()

    with h5py.File(h5_path, "r") as f:
        params = np.array([
            f["params/m1"][sample_id],
            f["params/m2"][sample_id],
            f["params/L1"][sample_id],
            f["params/L2"][sample_id],
            f["params/b1"][sample_id],
            f["params/b2"][sample_id],
            f["params/theta1_0"][sample_id],
            f["params/theta2_0"][sample_id],
        ], dtype=np.float32)

        th1 = f["trajectories/theta1"][sample_id]
        th2 = f["trajectories/theta2"][sample_id]
        dth1 = f["trajectories/dtheta1"][sample_id]
        dth2 = f["trajectories/dtheta2"][sample_id]

    T = len(th1)
    steps = min(steps, T-1)

    #(81,4)
    gt = np.stack([th1[:steps+1], th2[:steps+1], dth1[:steps+1], dth2[:steps+1]], axis=1)

    state = gt[0].copy()
    pred_traj = [state.copy()]

    for k in range(steps):
        x = np.concatenate([params, state], axis=0)[None, :]
        x_s = x_scaler.transform(x)
        xb = torch.tensor(x_s, dtype=torch.float32, device=device)
        with torch.no_grad():
            yb = model(xb).cpu().numpy()
        y_s = yb
        delta = y_scaler.inverse_transform(y_s)[0]
        state = state + delta
        pred_traj.append(state.copy())

    pred_traj = np.array(pred_traj)

    t = np.arange(steps+1)
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(t, gt[:,0], label="theta1 true")
    plt.plot(t, pred_traj[:,0], "--", label="theta1 pred")
    plt.plot(t, gt[:,1], label="theta2 true")
    plt.plot(t, pred_traj[:,1], "--", label="theta2 pred")
    plt.legend(); plt.grid(True); plt.title("Angles rollout")

    plt.subplot(2,1,2)
    plt.plot(t, gt[:,2], label="dtheta1 true")
    plt.plot(t, pred_traj[:,2], "--", label="dtheta1 pred")
    plt.plot(t, gt[:,3], label="dtheta2 true")
    plt.plot(t, pred_traj[:,3], "--", label="dtheta2 pred")
    plt.legend(); plt.grid(True); plt.title("Angular velocities rollout")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(sample_id=0, steps=80)