# K-Means con Iris Species

## Objetivo

Implementar un caso práctico de **K-Means** para agrupar flores Iris según sus características.

A diferencia de la regresión logística, K-Means es un algoritmo de **aprendizaje no supervisado**, por lo que no usa la columna `Species` para entrenar. El modelo solo utiliza las medidas de las flores para formar grupos.

---

## 1. Importar librerías

Primero importamos las librerías necesarias para trabajar con datos, escalar variables, entrenar el modelo y evaluar los grupos.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

---

## 2. Cargar el dataset

Cargamos el archivo CSV del dataset Iris desde Kaggle.

Dataset: https://www.kaggle.com/datasets/uciml/iris/data

```python
data = pd.read_csv("/kaggle/input/iris/Iris.csv")
```

---

## 3. Seleccionar variables

Seleccionamos las variables de entrada `X`.

En K-Means no usamos `y` para entrenar, porque no hay una variable objetivo.

```python
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
```

---

## 4. Escalar las variables

Escalamos las variables para que todas tengan una escala similar.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

## 5. Crear y entrenar el modelo K-Means

Creamos el modelo K-Means indicando que queremos formar 3 grupos, porque el dataset Iris tiene 3 especies.

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)
```

---

## 6. Obtener los clusters

Asignamos a cada flor el grupo o cluster que el modelo encontró.

```python
clusters = kmeans.labels_

data["Cluster"] = clusters

data.head()
```

---

## 7. Evaluar el modelo

En K-Means no usamos Accuracy, Precision, Recall ni F1-score para entrenar, porque no hay etiquetas reales durante el aprendizaje.

Usamos métricas propias de clustering como **Inertia** y **Silhouette Score**.

```python
inertia = kmeans.inertia_
silhouette = silhouette_score(X_scaled, clusters)

print("Inertia:", inertia)
print("Silhouette Score:", silhouette)
```

---

## 8. Visualizar los clusters

Graficamos los grupos usando dos variables: largo y ancho del pétalo.

```python
plt.scatter(
    data["PetalLengthCm"],
    data["PetalWidthCm"],
    c=data["Cluster"]
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Clusters encontrados con K-Means")
plt.show()
```

---

## 9. Comparar clusters con especies reales

Aunque K-Means no usa la columna `Species` para entrenar, podemos compararla después para observar cómo quedaron agrupadas las flores.

```python
comparison = pd.crosstab(data["Species"], data["Cluster"])

print(comparison)
```

---

## 10. Predecir el cluster de una nueva flor

Creamos una nueva flor y usamos el modelo para asignarla a un cluster.

```python
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
)

new_flower_scaled = scaler.transform(new_flower)

cluster_prediction = kmeans.predict(new_flower_scaled)

print("Cluster asignado:", cluster_prediction[0])
```

---

## Resultado esperado

El modelo debe agrupar las flores Iris en 3 clusters.

Como K-Means es no supervisado, el resultado no será directamente `Iris-setosa`, `Iris-versicolor` o `Iris-virginica`, sino grupos numéricos como:

* `Cluster 0`
* `Cluster 1`
* `Cluster 2`

Luego esos clusters pueden compararse con la columna `Species` para analizar si los grupos encontrados se parecen a las especies reales.

