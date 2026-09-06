import numpy as np
import h5py
import matplotlib.pyplot as plt
from new_simulator import PendulumSimulator

def get_true_trajectory(h5_path="pendulum_dataset.h5", trajectory_index=0):
    """Извлечь одну реальную траекторию из датасета"""
    with h5py.File(h5_path, "r") as f:
        th1 = f["trajectories/theta1"][trajectory_index]
        th2 = f["trajectories/theta2"][trajectory_index]
        dth1 = f["trajectories/dtheta1"][trajectory_index]
        dth2 = f["trajectories/dtheta2"][trajectory_index]
        
        params = np.array([
            f["params/m1"][trajectory_index],
            f["params/m2"][trajectory_index],
            f["params/L1"][trajectory_index],
            f["params/L2"][trajectory_index],
            f["params/b1"][trajectory_index],
            f["params/b2"][trajectory_index],
            f["params/theta1_0"][trajectory_index],
            f["params/theta2_0"][trajectory_index],
        ])
    
    trajectory = np.stack([th1, th2, dth1, dth2], axis=1)
    return trajectory, params

def run_autoregressive_simulation(simulator, true_trajectory, dt=0.01, max_steps=500):
    """
    Запускает авторегрессивную симуляцию и сравнивает с реальной траекторией.
    Возвращает ошибки по каждому шагу.
    """
    n_steps = min(max_steps, len(true_trajectory) - 1)
    
    predicted_states = [simulator.state.copy()]
    errors = []
    
    for t in range(n_steps):
        # Шаг симуляции
        predicted_state = simulator.step(dt)
        predicted_states.append(predicted_state)
        
        # Реальное состояние в момент t+1
        true_state = true_trajectory[t + 1]
        
        # Ошибка
        mse = np.mean((predicted_state - true_state)**2)
        errors.append(mse)
    
    return np.array(predicted_states), np.array(errors)

def main():
    # Загружаем одну траекторию из датасета
    true_traj, params = get_true_trajectory("pendulum_dataset.h5", trajectory_index=0)
    dt = 0.01  # шаг в датасете
    
    # Создаём симулятор
    sim = PendulumSimulator(
        model_path="mlp_best.pt",
        x_scaler_path="x_scaler.pkl",
        y_scaler_path="y_scaler.pkl",
        dt_train=dt
    )
    
    # Устанавливаем начальное состояние и параметры
    initial_state = true_traj[0]
    sim.set_initial_state(initial_state, params)
    
    # Запускаем авторегрессию
    predicted_traj, errors = run_autoregressive_simulation(sim, true_traj, dt, max_steps=500)
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    time = np.arange(len(predicted_traj)) * dt
    
    # Угол 1
    axes[0, 0].plot(time, true_traj[:, 0], 'b-', label='Реальный')
    axes[0, 0].plot(time, predicted_traj[:, 0], 'r--', label='Предсказанный')
    axes[0, 0].set_ylabel('θ1 (рад)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Угол 2
    axes[0, 1].plot(time, true_traj[:, 1], 'b-', label='Реальный')
    axes[0, 1].plot(time, predicted_traj[:, 1], 'r--', label='Предсказанный')
    axes[0, 1].set_ylabel('θ2 (рад)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Ошибка по времени
    axes[1, 0].plot(time[1:], errors)
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_xlabel('Время (с)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=0.1, color='r', linestyle='--', label='Порог 0.1')
    axes[1, 0].legend()
    
    # Фазовый портрет (ω1 от θ1)
    axes[1, 1].plot(true_traj[:, 0], true_traj[:, 2], 'b-', alpha=0.7, label='Реальный')
    axes[1, 1].plot(predicted_traj[:, 0], predicted_traj[:, 2], 'r--', alpha=0.7, label='Предсказанный')
    axes[1, 1].set_xlabel('θ1 (рад)')
    axes[1, 1].set_ylabel('ω1 (рад/с)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("error_analysis.png", dpi=150)
    plt.show()
    
    # Вывод статистики
    time_to_divergence = np.argmax(errors > 0.1) * dt
    print(f"Средняя одношаговая ошибка: {np.mean(errors):.6f}")
    print(f"Время до ошибки >0.1: {time_to_divergence:.2f} сек")
    print(f"Финальная ошибка после {len(errors)} шагов: {errors[-1]:.6f}")

if __name__ == "__main__":
    main()
