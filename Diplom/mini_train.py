import torch
from torch import nn

torch.manual_seed(0)

# 1) Делаем игрушечные данные
# X: (N, 12), Y: (N, 4)
N = 5000
X = torch.randn(N, 12)

# "Правда": пусть Y линейно зависит от X + чуть шум
W_true = torch.randn(12, 4)
Y = X @ W_true + 0.05 * torch.randn(N, 4)

# 2) Простая модель (MLP)
model = nn.Sequential(
    nn.Linear(12, 64),
    nn.Tanh(),
    nn.Linear(64, 4),
)

# 3) Loss и optimizer
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4) Цикл обучения
for epoch in range(1, 21):
    pred = model(X)              # forward
    loss = loss_fn(pred, Y)      # считаем ошибку

    opt.zero_grad()              # обнуляем градиенты
    loss.backward()              # считаем градиенты
    opt.step()                   # делаем шаг оптимизации

    if epoch % 2 == 0:
        print(f"epoch {epoch:02d}: mse={loss.item():.6f}")

print("done")