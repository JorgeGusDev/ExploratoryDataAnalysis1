> **Nota metodológica**
>
> El proceso de identificación y tratamiento de valores faltantes utilizado en este análisis sigue las recomendaciones generales de preprocesamiento de datos presentadas por Géron (2022) en la etapa *Prepare the Data for Machine Learning Algorithms*, donde se discuten estrategias como la eliminación de atributos, eliminación de instancias e imputación de valores faltantes mediante medidas estadísticas apropiadas.
>
> **Criterios prácticos aplicados en este proyecto**
>
> | Porcentaje de valores faltantes | Acción sugerida |
> |---------------------------------|----------------|
> | Menor al 5% | Eliminar registros o imputar valores |
> | Entre 5% y 30% | Imputar utilizando media, mediana o moda |
> | Mayor al 50% | Evaluar la eliminación de la variable |
>
> Estos umbrales constituyen reglas prácticas frecuentemente utilizadas en proyectos de Ciencia de Datos y Machine Learning para facilitar la toma de decisiones durante la limpieza de datos. La selección final debe considerar el contexto del problema y la importancia de cada variable.
>
> **Referencia**
>
> Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. Capítulo 2: *End-to-End Machine Learning Project*, sección *Prepare the Data for Machine Learning Algorithms*.
