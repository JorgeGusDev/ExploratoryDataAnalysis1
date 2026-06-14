## Tratamiento de Valores Faltantes

Antes de aplicar técnicas de análisis exploratorio y modelos de Machine Learning, es necesario evaluar la calidad de los datos e identificar la presencia de valores faltantes. De acuerdo con Géron (2022), durante la etapa de preparación de datos para algoritmos de aprendizaje automático pueden emplearse diversas estrategias, incluyendo la eliminación de atributos, la eliminación de instancias y la imputación de valores mediante estadísticas descriptivas.

En este análisis se emplearon los siguientes criterios prácticos para el tratamiento de valores faltantes:

| Porcentaje de valores faltantes | Acción recomendada                       |
| ------------------------------- | ---------------------------------------- |
| Menor al 5%                     | Eliminar registros o imputar valores     |
| Entre 5% y 30%                  | Imputar utilizando media, mediana o moda |
| Mayor al 50%                    | Evaluar la eliminación de la variable    |

Estos criterios constituyen una guía práctica para la limpieza de datos y deben complementarse con el análisis del contexto y la relevancia de cada variable para el problema estudiado.

### Referencia

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. Capítulo 2: *End-to-End Machine Learning Project*, sección *Prepare the Data for Machine Learning Algorithms*.
