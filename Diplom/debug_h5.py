import os
import h5py

filename = "pendulum_dataset.h5"

print("cwd:", os.getcwd())
print("exists:", os.path.exists(filename))
if os.path.exists(filename):
    print("size bytes:", os.path.getsize(filename))

with h5py.File(filename, "r") as f:
    print("root keys:", list(f.keys()))
    for k in f.keys():
        print(k, "->", list(f[k].keys()))
    print("theta1 shape:", f["trajectories/theta1"].shape)
    print("m1[0]:", f["params/m1"][0])