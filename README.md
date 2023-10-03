![](https://blog.soyhenry.com/content/images/2021/05/PRESENTACION-3.jpg)
# Primer Proyecto Individual
## Machine Learning Operations (MLOps)

El proyecto se plantea a partir de 3 data sets con información de video juegos de la pagina steam, al cual se le realizan un trabajo de Data Engineering haciendo una serie de transformaciones para luego llevar a cabo  los endpoints pedidos y un modelo de recomendación de videjuegos utilizando Machine Learning, a través de una API.

![](https://earthweb.com/wp-content/uploads/2022/05/Steam-940.jpg)

### Dataset:

- steam_game.csv
- user_reviews.csv
- user_items.csv
  
Los tres datasets  poseen información acerca de videojuegos y distintos atributos de los mismos. El primero, steam_games contaba con 120445 filas y 13 columnas, el segundo user_reviews contaba con 25799 filas y 3 columnas de las cuales una estaba empaquetada y el último user_items.csv con 88310 datos y 5 columnas una de ellas anidada y que al desanidarla llevan al dataset a tener 5153209 datos.todo esto se puede ver en el respectivo archivo: 

### Diccionario de Datos:

#### steam_game.csv
  
- User_Id: Esta columna parece ser un identificador de usuario o jugador.

- App_Name: Representa el nombre de una aplicación o juego.

- Release_Date: Muestra la fecha de lanzamiento de la aplicación o juego.

- Release_Year: Indica el año de lanzamiento de la aplicación o juego. Parece ser una versión resumida de la fecha de lanzamiento.

- Price: Indica el precio de la aplicación o juego.

- Genres: Describe los géneros asociados a la aplicación o juego, separados por comas.

- Specs: Muestra especificaciones técnicas o características de la aplicación o juego, separadas por comas.

- Early_Access: Un valor booleano que indica si la aplicación o juego está disponible en acceso anticipado (Early Access). True significa que está en acceso anticipado, False significa que no lo está.

- Developer: Representa el desarrollador o creador de la aplicación o juego.

- Publisher: Indica la editorial o empresa que publicó la aplicación o juego.

#### user_reviews.csv

- User_Id: Un identificador único para cada usuario.
  
- Item_Id: Un identificador único para cada juego.
  
- Item_Name: El nombre del juego.
  
- Playtime_Forever: La cantidad de tiempo que un usuario ha jugado a un juego en minutos.

#### user_id.csv

- User_Id: Representa el identificador único de usuario.
  
- Posted: Indica la fecha en que se realizó una publicación o revisión.
  
- Item_Id: Es un identificador único para un juego.
  
- Recommend: Es un valor booleano que indica si la revisión recomienda o no el juego.
  
- Year: Muestra el año en que se realizó la revisión.

- Sentiment_Analysis: Es un valor numérico que  representa un análisis de sentimiento de la valoración del juego.

### Librerias y herramientas utilizadas en el proyecto:

- Scikit Learn: Utilizado para vectorizar, tokenizar y calcular la similitud coseno.
- Python: Lenguaje de programación principal utilizado en el desarrollo del proyecto.
- Numpy: Utilizado para realizar operaciones numéricas y manipulación de datos.
- Pandas: Utilizado para la manipulación y análisis de datos estructurados.
- Matplotlib: Utilizado para la visualización de datos y generación de gráficos.
- FastAPI: Utilizado para crear la interfaz de la aplicación y procesar los parámetros de funciones.
- Uvicorn: Servidor ASGI utilizado para ejecutar la aplicación FastAPI.
- Render: Plataforma utilizada para el despliegue del modelo y la aplicación.

## DESARROLLO DEL PROYECTO:

### Data Engineering:

En el ámbito de la Ingeniería de Datos, se llevó a cabo un conjunto de transformaciones que se requerian para poder proesar el datasets para realizar la api luego. Se buscaron nulos, se cambiaron y normalizaron los nombres de columnas, se eliminaron columnas y filas que no se iban a ocupar, se buscaron repetidos y se busco ordenar las columnas para que el dataset a la vista se lea mejor se vea mejor. También se creo una columna 'sentiment_analysis' aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna reemplazó la de user_reviews.review  

Se pueden visualizar las transformaciones y los análisis realizados en el siguiente archivo: [ETL.ipynb](https://github.com/maga86/Proyecto_Individual-MLops/blob/main/ETL.ipynb)

### Desarrollo API:

Se disponibilizó los datos de la empresa usando el framework FastAPI. Las consultas fueron las siguientes:

1- def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
Ejemplo de retorno: {"Año de lanzamiento con más horas jugadas para Género X" : 2013}

2- def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}

3- def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

4- def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]

5- def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278

Se pueden visualizar el desarrollo de las funciones en los siguientes archivos:
- [Endspoints.ipynb](https://github.com/maga86/Henry-1-Proyecto-Individual-MLops/blob/main/Endspoints.ipynb)
- [main.py](https://github.com/maga86/Henry-1-Proyecto-Individual-MLops/blob/main/main.py)

### Modelo de recomendación:
![](https://www.go4it.solutions/sites/default/files/2021-06/05.01.%20Qu%C3%A9%20es%20el%20Machine%20Learning.jpg)
Se pidió un modelo que  deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar es ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados, para ello recomendamos aplicar la similitud del coseno. Se pide que el modelo derive obligatoriamente en un GET/POST en la API símil al siguiente formato:

Si es un sistema de recomendación item-item:

def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Se realizo el sistema de recomendación utilizando las columnas app_name, item_name, Genres,Specs y Id_Items si bien se aconseja el modelo con la similitud del coseno, lo terminé realizando con K vecinos por una cuestion de recursos de mi máquina.Se pueden ver en los siguientes archivos: 
- [Sistema de Recomendación.ipynb](https://github.com/maga86/Henry-1-Proyecto-Individual-MLops/blob/main/Sistema%20de%20Recomendaci%C3%B3n.ipynb)
- [main.py](https://github.com/maga86/Henry-1-Proyecto-Individual-MLops/blob/main/main.py)


### Análisis Exploratorio de Datos:

Se realizaron una serie de análisis y estudios sobre las variables del dataset  para  poder encontrar relaciones entre los datos y comprender la relevancia de los mismos. Dentro de los análisis efectuados se encuentran gráficos de palabras, gráficos de barras comparando columnas, distribuciones de frecuencias de las variables numéricas, identificación de variables categóricas y sus valores, correlación entre variables, detección de outliers, análisis temporales y por categoría,comparación entre columnas.

Se puede visualizar el Análisis exploratorio en el archivo:[EDA.ipynb](https://github.com/maga86/Henry-1-Proyecto-Individual-MLops/blob/main/EDA.ipynb)

### Deploy:

Finalizando se llevo acabo el deploy de la apirest en la pagina render. Se puede ver el funcionamiento de la api en el siguiente enlace: https://steam-api-rest.onrender.com/docs
