import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(filename="pendulum_dataset.h5"):
    
    with h5py.File(filename, 'r') as f:
        # Достаем матрицы траекторий
        theta1 = f['trajectories/theta1'][:]
        theta2 = f['trajectories/theta2'][:]
        dtheta1 = f['trajectories/dtheta1'][:]
        dtheta2 = f['trajectories/dtheta2'][:]
        

    print("Данные загружены.")
    
    df = pd.DataFrame({
        'theta1': theta1.flatten(),
        'theta2': theta2.flatten(),
        'dtheta1': dtheta1.flatten(),
        'dtheta2': dtheta2.flatten()
    })

    print("\n1. ОБЩАЯ СТАТИСТИКА")
    
    print(df.describe().round(3)) 

    print("\nПроверка на пустые значения (NaN):")
    print(df.isna().sum())

    #thr = 200
    #print("Сколько |dtheta1| > 200:", (np.abs(df["dtheta1"]) > thr).sum())
    #print("Сколько |dtheta2| > 200:", (np.abs(df["dtheta2"]) > thr).sum())
    #print("Доля |dtheta1| > 200:", (np.abs(df["dtheta1"]) > thr).mean())

    
    print("\nСтроим графики")
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Анализ распределения данных (Проверка на выбросы)', fontsize=16)

    sns.histplot(df['theta1'], bins=100, ax=axes[0,0], color='blue').set_title('Угол 1 (theta1)')
    sns.histplot(df['theta2'], bins=100, ax=axes[0,1], color='orange').set_title('Угол 2 (theta2)')
    sns.histplot(df['dtheta1'], bins=100, ax=axes[1,0], color='green').set_title('Скорость 1 (dtheta1)')
    sns.histplot(df['dtheta2'], bins=100, ax=axes[1,1], color='red').set_title('Скорость 2 (dtheta2)')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['dtheta1', 'dtheta2']])
    plt.title('Поиск аномалий в скоростях')
    plt.show()

if __name__ == "__main__":
    analyze_dataset()