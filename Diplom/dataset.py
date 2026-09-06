import numpy as np
import h5py
import itertools
from tqdm import tqdm
from simulator import simulate

def generate_dataset(filename="boom.h5"):
    
    grid = {
        "m1": np.linspace(0.5, 2.0, 3),
        "m2": np.linspace(0.5, 2.0, 3),
        "L1": np.linspace(0.3, 0.9, 3),
        "L2": np.linspace(0.3, 0.9, 3),

        "b1": np.logspace(-2, -1, 2),
        "b2": np.logspace(-2, -1, 2),

        "theta1_0": np.linspace(0.5, 2.5, 3),
        "theta2_0": np.linspace(0.5, 2.5, 3),
    }

    t_end = 5.0
    dt = 0.05
    n_steps = int(t_end / dt)

    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    num_samples = len(combinations)
    
    print(f"Всего комбинаций параметров: {num_samples}")
    print(f"Точек в каждой траектории: {n_steps}")
    print(f"Создаем файл {filename}...")

    with h5py.File(filename, 'w') as f:

        grp_params = f.create_group('params')
        grp_traj = f.create_group('trajectories')

        for key in keys:
            grp_params.create_dataset(key, shape=(num_samples,), dtype='float32')
        
        ds_t = grp_traj.create_dataset('t', shape=(n_steps,), dtype='float32')
        ds_theta1 = grp_traj.create_dataset('theta1', shape=(num_samples, n_steps), dtype='float32', compression="gzip")
        ds_theta2 = grp_traj.create_dataset('theta2', shape=(num_samples, n_steps), dtype='float32', compression="gzip")
        ds_dtheta1 = grp_traj.create_dataset('dtheta1', shape=(num_samples, n_steps), dtype='float32', compression="gzip")
        ds_dtheta2 = grp_traj.create_dataset('dtheta2', shape=(num_samples, n_steps), dtype='float32', compression="gzip")

        time_saved = False

        for i, params_tuple in enumerate(tqdm(combinations, desc="Симуляция")):

            p = dict(zip(keys, params_tuple))
            
            result = simulate(
                m1=p['m1'], m2=p['m2'], 
                L1=p['L1'], L2=p['L2'], 
                b1=p['b1'], b2=p['b2'],
                theta1_0=p['theta1_0'], theta2_0=p['theta2_0'],
                dtheta1_0=0.0, dtheta2_0=0.0,
                t_end=t_end, dt=dt
            )

            for key in keys:
                grp_params[key][i] = p[key]

            if not time_saved:
                ds_t[:] = result['t']
                time_saved = True

            ds_theta1[i, :] = result['theta1']
            ds_theta2[i, :] = result['theta2']
            ds_dtheta1[i, :] = result['dtheta1']
            ds_dtheta2[i, :] = result['dtheta2']

    print(f"\nДатасет успешно сохранен в {filename}")
    
    #Проверка
    with h5py.File(filename, 'r') as f:
        print("\nПроверка сохраненного файла:")
        print(f"Форма theta1: {f['trajectories/theta1'].shape}")
        print(f"Параметр m1 для первой траектории: {f['params/m1'][0]:.2f}")

if __name__ == "__main__":
    generate_dataset()