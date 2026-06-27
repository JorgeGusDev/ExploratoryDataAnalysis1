# Caso Práctico usando Python # 2

##  Preprocesamiento de los Datos
El preprocesamiento de datos es una etapa crítica en proyectos de Inteligencia Artificial y Machine Learning, ya que permite transformar los datos crudos en un formato adecuado para el modelado. Las principales fases incluyen: Limpieza de datos, Tratamiento de valores nulos, Transformación de variables, Reducción de ruido e inconsistencias, Escalado y normalización


---

## Tratamiento de Datos Nulos

Los valores nulos (missing values) pueden afectar negativamente el rendimiento de los modelos, por lo que deben ser tratados adecuadamente.

### Eliminación de datos nulos

Se eliminan filas o columnas que contienen valores faltantes.

```python
df_clean = df.dropna()
```

**Uso:** Cuando la cantidad de datos nulos es pequeña.

---

### Reemplazo con media, mediana o moda

```python
# Media
df["variable"].fillna(df["variable"].mean(), inplace=True)

# Mediana
df["variable"].fillna(df["variable"].median(), inplace=True)

# Moda
df["variable"].fillna(df["variable"].mode()[0], inplace=True)
```

**Uso:**

* Media → datos numéricos simétricos
* Mediana → datos con outliers
* Moda → variables categóricas

---

### Asignación a una categoría única

```python
df["variable"].fillna("Desconocido", inplace=True)
```

**Uso:** Variables categóricas donde el valor faltante representa una categoría válida.

---

### Predicción de valores faltantes por regresión

Se utilizan modelos predictivos para estimar valores ausentes.

```python
from sklearn.linear_model import LinearRegression

# Ejemplo conceptual
model = LinearRegression()
model.fit(X_train, y_train)

df.loc[df["variable"].isnull(), "variable"] = model.predict(X_missing)
```

**Uso:** Cuando los datos tienen relaciones fuertes entre variables.

---

## Escalado de Características en Machine Learning

El escalado normaliza las variables para que tengan la misma magnitud, evitando que algunas dominen el modelo.

### Normalización (Min-Max Scaling)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["variable"]])
```

**Rango:** 0 a 1


---

## Referencia

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
Capítulo 2: *End-to-End Machine Learning Project*.

---
