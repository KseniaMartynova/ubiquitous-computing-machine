import matplotlib.pyplot as plt
import numpy as np

# Данные для построения графика
values = [0.000782807, 0.000869859, 0.000840406, 0.000829538, 0.000774271, 
          0.000755921, 0.000829167, 0.000798261, 0.000833738, 0.000858666]

# Создание массива индексов для оси X
indices = np.arange(len(values))

# Создание графика
fig, ax = plt.subplots()
ax.plot(indices, values, marker='o')

# Настройка графика
ax.set(xlabel='Number of Points', ylabel='Секунды',
       title='armadillo 100 на 100')
ax.grid()

# Установка значений на оси X
ax.set_xticks(indices)

# Добавление подписей к точкам
for i, value in zip(indices, values):
    ax.text(i, value, f'{value:.7f}', ha='center', va='bottom')

# Сохранение графика в файл
fig.savefig("values_plot.png")

# Отображение графика
plt.show()
