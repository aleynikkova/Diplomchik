import numpy as np
import torch
import joblib
from train_mlp import MLP

class PendulumSimulator:
    def __init__(self, model_path="mlp_best.pt", 
                 x_scaler_path="x_scaler.pkl",
                 y_scaler_path="y_scaler.pkl",
                 dt_train=0.01):
        """
        dt_train - шаг времени, на котором обучалась сеть (из HDF5)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем модель
        self.model = MLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Загружаем нормализаторы
        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)
        
        self.dt_train = dt_train  # шаг, на котором училась сеть
        self.state = None  # текущее состояние [θ1, θ2, ω1, ω2]
        self.params = None  # параметры маятника [m1, m2, L1, L2, b1, b2, θ1_0, θ2_0]
        
    def set_initial_state(self, state, params):
        """Установить начальное состояние и параметры"""
        self.state = np.array(state, dtype=np.float32)
        self.params = np.array(params, dtype=np.float32)
        
    def step(self, dt_real):
        """
        Сделать шаг симуляции на dt_real секунд.
        Возвращает новое состояние.
        """
        if self.state is None or self.params is None:
            raise ValueError("Сначала вызови set_initial_state()")
        
        # Нормируем количество шагов (если dt_real > dt_train, делаем несколько шагов)
        n_steps = max(1, int(dt_real / self.dt_train))
        effective_dt = dt_real / n_steps
        
        for _ in range(n_steps):
            self._step_internal(effective_dt)
        
        return self.state.copy()
    
    def _step_internal(self, dt):
        """Один шаг внутренней симуляции"""
        # 1. Собираем входной вектор [params, state]
        X = np.concatenate([self.params, self.state])
        
        # 2. Нормализуем
        X_scaled = self.x_scaler.transform([X])
        
        # 3. Предсказываем производную
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            dstate_scaled = self.model(X_tensor).cpu().numpy()[0]
        
        # 4. Денормализуем производную
        dstate = self.y_scaler.inverse_transform([dstate_scaled])[0]
        
        # 5. Обновляем состояние (метод Эйлера)
        self.state = self.state + dstate * dt
        
        # 6. Опционально: ограничиваем углы (но не обязательно для динамики)
        # self.state[0] = self.state[0] % (2*np.pi)  # θ1 в [0, 2π]
        # self.state[1] = self.state[1] % (2*np.pi)  # θ2 в [0, 2π]
        
    def get_state(self):
        return self.state.copy()
