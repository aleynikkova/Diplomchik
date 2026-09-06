import h5py
import numpy as np

file_path = '/Users/macpro2021/Desktop/Diplom/pendulum_dataset.h5'

with h5py.File(file_path, 'r') as f:
    print("="*80)
    print(f"Файл: {file_path}")
    print("="*80)
    
    def show_all_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"\n{'─'*60}")
            print(f"📊 Имя: {name}")
            print(f"   Форма (shape): {obj.shape}")
            print(f"   Тип данных: {obj.dtype}")
            
            # Загружаем все данные
            data = obj[()]  # Полный массив
            
            # Разные варианты вывода в зависимости от размера
            if obj.size == 0:
                print("   [пустой датасет]")
                
            elif obj.size <= 1000:
                # Если данных мало - показываем полностью
                print(f"\n   Содержимое:\n{data}")
                
            elif obj.ndim == 1:
                # 1D массив (вектор)
                print(f"\n   Первые 100 значений: {data[:100]}")
                print(f"   Последние 10 значений: {data[-10:]}")
                print(f"   Статистика: мин={data.min():.4f}, макс={data.max():.4f}, среднее={data.mean():.4f}")
                
            elif obj.ndim == 2:
                # 2D массив (матрица) - скорее всего ваши траектории
                print(f"\n   Размер: {obj.shape[0]} строк × {obj.shape[1]} столбцов")
                print(f"   Первые 5 строк (первые 5 траекторий):")
                for i in range(min(5, obj.shape[0])):
                    print(f"     Траектория {i}: {data[i, :10]}...")  # первые 10 точек
                print(f"   Последняя траектория: {data[-1, :10]}...")
                
            else:
                # Многомерные массивы
                print(f"\n   Первые 100 элементов (в развернутом виде): {data.flatten()[:100]}")
            
            print(f"   Размер в памяти: {obj.nbytes / 1024:.2f} KB")
    
    # Обходим все датасеты в файле
    f.visititems(show_all_data)
    
    # Дополнительно: выгружаем всё в переменные для работы
    print("\n" + "="*80)
    print("📦 Загружаем все данные в словарь для дальнейшей работы:")
    all_data = {}
    
    for name, obj in f.items():
        if isinstance(obj, h5py.Dataset):
            all_data[name] = obj[()]
            print(f"   {name}: форма {obj.shape}")
    
    # Если хотите поработать с конкретным массивом траекторий
    # Например, если массив называется 'trajectories' или 'data'
    # trajectories = all_data.get('trajectories', all_data.get('data', None))