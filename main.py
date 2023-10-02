# Importamos las librerías
from fastapi import FastAPI
import pandas as pd
import numpy as np
import ast
import re
from sklearn.neighbors import NearestNeighbors

# Indicamos título y descripción de la API
app = FastAPI(title='PROYECTO INDIVIDUAL Nº1 ',
            description='Steam Games')

# Dataset
df = pd.read_csv('steam_games.csv')
df2 = pd.read_csv('user_reviews.csv')
df3 = pd.read_csv('user_items.csv')


@app.get('/Play_Time_Genre/{Genres}')
def PlayTimeGenre(genero: str):
    
    '''Debe devolver año con mas horas jugadas para dicho género.'''
    
    df['user_id'] = df['user_id'].astype(str)
    df3['user_id'] = df3['user_id'].astype(str)
    # Fusionar ambos DataFrames en base a la columna 'user_id'
    df_combinado = df.merge(df3, left_on='user_id', right_on='user_id', how='inner')
    df_filtrado = df_combinado[df_combinado['Genres'].str.contains(genero, case=False, na=False)]

    if df_filtrado.empty:
        return {}  

    agrupado = df_filtrado.groupby('Release_Year')['playtime_forever'].sum()
    max_year = agrupado.idxmax()
    
    # Convertir max_year a int antes de devolverlo
    max_year = int(max_year)
    
    resultado = {"Año de lanzamiento con más horas jugadas para Género " + genero: max_year}

    return resultado

@app.get('/User_For_Genre/{Genres}')
def UserForGenre(genero: str):
    
    ''' Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.'''
    
    df['user_id'] = df['user_id'].astype(str)
    df3['user_id'] = df3['user_id'].astype(str)
    
    # Fusionar ambos DataFrames en base a la columna 'user_id'
    df_combinado = df.merge(df3, left_on='user_id', right_on='user_id', how='inner')

    df_filtrado = df_combinado[df_combinado['Genres'].str.contains(genero, case=False, na=False)]
    if df_filtrado.empty:
        return {}  
    agrupado = df_filtrado.groupby(['user_id', 'Release_Year'])['playtime_forever'].sum().reset_index()
    max_user = agrupado.groupby('user_id')['playtime_forever'].sum().idxmax()
    df_max_user = agrupado[agrupado['user_id'] == max_user]
    lista_acumulacion = df_max_user.rename(columns={'Release_Year': 'Año', 'playtime_forever': 'Horas'}).to_dict('records')
    resultado = {
        "Usuario con más horas jugadas para el Género " + genero: max_user,
        "Horas jugadas": lista_acumulacion
    }

    return resultado

@app.get('/User_Recommend/{anio}')
def UsersRecommend(año: int):
    ''' Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)'''
    
    # filas para el año  y donde recommend es False
    juegos_no_recomendados = df2[(df2['recommend'] == True)]
    
    # muestra aleatoria de df3
    sample_size = 90000  
    df3_sample = df3.sample(n=sample_size, random_state=42)
    
    
    juegos_no_recomendados = juegos_no_recomendados.merge(df3_sample[['item_id', 'item_name']], on='item_id', how='left')
    conteo_juegos = juegos_no_recomendados['item_name'].value_counts().reset_index()
    conteo_juegos.columns = ['item_name', 'count']
    top_juegos_no_recomendados = conteo_juegos.sort_values(by='count', ascending=True).head(3)
    
    resultado = [{"Puesto {}: {}".format(i+1, juego)} for i, juego in enumerate(top_juegos_no_recomendados['item_name'])]
    
    return resultado

@app.get('/User_Not_Recommend/{anio}')
def UsersNotRecommend(año: int):
    
    '''Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.'''
    
     #filas para el año dado y donde recommend es False
    juegos_no_recomendados = df2[(df2['recommend'] == False)]
    
    juegos_no_recomendados = juegos_no_recomendados.merge(df3[['item_id', 'item_name']], on='item_id', how='left')
    conteo_juegos = juegos_no_recomendados['item_name'].value_counts().reset_index()
    conteo_juegos.columns = ['item_name', 'count']
    
    top_juegos_no_recomendados = conteo_juegos.sort_values(by='count', ascending=True).head(3)
    
    
    resultado = [{"Puesto {}: {}".format(i+1, juego)} for i, juego in enumerate(top_juegos_no_recomendados['item_name'])]
    
    return resultado

@app.get('/Sentiment_Analysis/{anio}')
def sentiment_analysis(anio: int):
    
    ''' Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento'''
    
    # DataFrame para obtener solo las filas correspondientes al año especificado
    df_filtrado = df2[df2['Year'] == anio]
    categorias_sentimiento = df_filtrado['sentiment_analysis'].value_counts()
    
    # Convertir la Serie en un diccionario
    categorias_sentimiento_dict = categorias_sentimiento.to_dict()
    
    resultado = {
        'Negativo': categorias_sentimiento_dict.get(0, 0),
        'Neutral': categorias_sentimiento_dict.get(1, 0),
        'Positivo': categorias_sentimiento_dict.get(2, 0)
    }

    return resultado



# Cargar los datos del DataFrame
df = pd.read_csv('steam_games.csv')

#  características categóricas
genres_dummies = df['Genres'].str.get_dummies(sep=', ')
specs_dummies = df['Specs'].str.get_dummies(sep=', ')
features_matrix = pd.concat([genres_dummies, specs_dummies], axis=1)


# Entrenar un modelo K-NN
k = 5  # Número de vecinos cercanos a considerar
model = NearestNeighbors(n_neighbors=k, metric='cosine')
model.fit(features_matrix)

def get_recommendations(game_id, num_recommendations=5):
    game_index = df[df['App_Name'] == game_id].index[0]
    distances, indices = model.kneighbors([features_matrix.iloc[game_index]], n_neighbors=num_recommendations + 1)
    recommended_game_indices = indices[0][1:]  # Excluye el propio juego
    recommended_games = df.iloc[recommended_game_indices]['App_Name'].tolist()
    return recommended_games

@app.get("/recommendations/{app_name}")
def get_recommendations_endpoint(app_name: str, num_recommendations: int = 5):
    
    '''Se obtienen 5 juegos de las mismas carateristicas que el ingresado'''
    
    recommended_games = get_recommendations(app_name, num_recommendations)
    return {"recommended_games": recommended_games}

