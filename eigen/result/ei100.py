import matplotlib.pyplot as plt
import numpy as np

# Данные для построения графика
values = [0.0183887, 0.0184044,0.0184076,0.0184081, 0.018416,0.0184364,  0.0184487, 0.0185227,0.018453, 0.0184943]

# Создание массива индексов для оси X
indices = np.arange(len(values))

# Создание графика
fig, ax = plt.subplots()
ax.plot(indices, values, marker='o', label='Values 1', color='red')


# Настройка графика
ax.set(xlabel='Number of Points', ylabel='Секунды',
       title='eigen 100 на 100')
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
