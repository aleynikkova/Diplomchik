import h5py
import numpy as np

from simulator import simulate


# Уровни трения:
# 0.01 — слабое
# 0.05 — среднее
# 0.1  — сильное
FRICTIONS = [0.01, 0.05, 0.1]

# Количество траекторий для каждого уровня трения
N_TRAJECTORIES = 1000

# Параметры моделирования
M1 = 1.0
M2 = 1.0
L1 = 1.0
L2 = 0.5

T_END = 15.0
DT = 0.01

RNG = np.random.default_rng(42)


def generate_dataset(filename, friction, n_trajectories):
    """Генерирует HDF5-датасет для заданного коэффициента трения."""

    theta1_0 = RNG.uniform(-1.5, 1.5, n_trajectories)
    theta2_0 = RNG.uniform(-1.5, 1.5, n_trajectories)

    trajectories = []

    for i in range(n_trajectories):
        result = simulate(
            m1=M1,
            m2=M2,
            L1=L1,
            L2=L2,
            b1=friction,
            b2=friction,
            theta1_0=theta1_0[i],
            theta2_0=theta2_0[i],
            t_end=T_END,
            dt=DT
        )

        trajectories.append(result)

    theta1 = np.array(
        [result["theta1"] for result in trajectories],
        dtype=np.float32
    )

    theta2 = np.array(
        [result["theta2"] for result in trajectories],
        dtype=np.float32
    )

    dtheta1 = np.array(
        [result["dtheta1"] for result in trajectories],
        dtype=np.float32
    )

    dtheta2 = np.array(
        [result["dtheta2"] for result in trajectories],
        dtype=np.float32
    )

    with h5py.File(filename, "w") as f:

        params = f.create_group("params")

        params.create_dataset(
            "m1",
            data=np.full(n_trajectories, M1, dtype=np.float32)
        )

        params.create_dataset(
            "m2",
            data=np.full(n_trajectories, M2, dtype=np.float32)
        )

        params.create_dataset(
            "L1",
            data=np.full(n_trajectories, L1, dtype=np.float32)
        )

        params.create_dataset(
            "L2",
            data=np.full(n_trajectories, L2, dtype=np.float32)
        )

        params.create_dataset(
            "b1",
            data=np.full(n_trajectories, friction, dtype=np.float32)
        )

        params.create_dataset(
            "b2",
            data=np.full(n_trajectories, friction, dtype=np.float32)
        )

        params.create_dataset(
            "theta1_0",
            data=theta1_0.astype(np.float32)
        )

        params.create_dataset(
            "theta2_0",
            data=theta2_0.astype(np.float32)
        )

        trajectories_group = f.create_group("trajectories")

        trajectories_group.create_dataset(
            "theta1",
            data=theta1
        )

        trajectories_group.create_dataset(
            "theta2",
            data=theta2
        )

        trajectories_group.create_dataset(
            "dtheta1",
            data=dtheta1
        )

        trajectories_group.create_dataset(
            "dtheta2",
            data=dtheta2
        )

    print(f"Создан датасет: {filename}")
    print(f"Трение: b1 = b2 = {friction}")
    print(f"Количество траекторий: {n_trajectories}")
    print(f"Размер траектории: {theta1.shape[1]}")


def main():
    for friction in FRICTIONS:

        filename = f"pendulum_friction_{friction:.2f}.h5"

        generate_dataset(
            filename=filename,
            friction=friction,
            n_trajectories=N_TRAJECTORIES
        )


if __name__ == "__main__":
    main()