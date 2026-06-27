# Regresión Lineal con Gradient Descent

## Objetivo

Implementar un caso práctico simple de **regresión lineal** usando **Gradient Descent** para predecir el salario de una persona según sus años de experiencia.

El dataset usado contiene dos columnas principales:

* `YearsExperience`
* `Salary`

---

## Idea del modelo

La regresión lineal busca encontrar una relación entre una variable de entrada y una variable de salida numérica.

En este caso:

* Entrada: años de experiencia
* Salida: salario

La fórmula del modelo es:

```text
y_pred = wX + b
```

Donde:

```text
X = años de experiencia
y_pred = salario predicho
w = peso o pendiente
b = sesgo o intercepto
```

---

## Gradient Descent

Gradient Descent es un algoritmo de optimización que ajusta poco a poco los valores de `w` y `b` para reducir el error del modelo.

La función de error usada es el **Mean Squared Error (MSE)**:

```text
MSE = promedio de (salario real - salario predicho)^2
```

El objetivo es que este error disminuya durante el entrenamiento.

---

## Código en Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar dataset
data = pd.read_csv("Salary_Data.csv")

# Seleccionar variables
X = data["YearsExperience"].values
y = data["Salary"].values

# Normalizar X para mejorar Gradient Descent
X_mean = X.mean()
X_std = X.std()
X_scaled = (X - X_mean) / X_std

# Inicializar parámetros
w = 0
b = 0

# Hiperparámetros
learning_rate = 0.01
epochs = 1000
n = len(X_scaled)

loss_history = []

# Entrenamiento con Gradient Descent
for epoch in range(epochs):
    # Predicción
    y_pred = w * X_scaled + b

    # Error
    error = y_pred - y

    # Calcular MSE
    mse = np.mean(error ** 2)
    loss_history.append(mse)

    # Calcular gradientes
    dw = (2 / n) * np.sum(error * X_scaled)
    db = (2 / n) * np.sum(error)

    # Actualizar parámetros
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Resultados finales
print("Peso final:", w)
print("Sesgo final:", b)
print("Error final:", loss_history[-1])

# Graficar datos reales y línea de regresión
plt.scatter(X, y, label="Datos reales")

X_line = np.linspace(X.min(), X.max(), 100)
X_line_scaled = (X_line - X_mean) / X_std
y_line = w * X_line_scaled + b

plt.plot(X_line, y_line, label="Línea de regresión")
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.title("Regresión Lineal con Gradient Descent")
plt.legend()
plt.show()

# Graficar disminución del error
plt.plot(loss_history)
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.title("Disminución del error")
plt.show()

# Realizar una predicción
years = 5
years_scaled = (years - X_mean) / X_std
salary_pred = w * years_scaled + b

print(f"Salario predicho para {years} años de experiencia: {salary_pred:.2f}")
```

---

## Resultado esperado

El modelo debe aprender una relación positiva entre los años de experiencia y el salario.

Esto significa que, mientras más años de experiencia tenga una persona, mayor debería ser el salario estimado por el modelo.

Además, el gráfico del error debe mostrar una disminución progresiva del **MSE**, lo que indica que Gradient Descent está funcionando correctamente.

---

## Conceptos clave

* La regresión lineal predice valores numéricos.
* Gradient Descent ajusta los parámetros del modelo.
* `w` representa la pendiente de la línea.
* `b` representa el intercepto o sesgo.
* `learning_rate` controla el tamaño de cada actualización.
* `epochs` indica cuántas veces se actualizan los parámetros.
* `MSE` mide el error entre el valor real y el valor predicho.

---

## Conclusión

Este caso práctico muestra cómo entrenar una regresión lineal desde cero usando Gradient Descent. El modelo aprende ajustando sus parámetros para reducir el error y encontrar una línea que represente la relación entre experiencia laboral y salario.
