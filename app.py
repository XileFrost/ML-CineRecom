# Importaciones necesarias
import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
import difflib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

# Configuración de la página
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar y preprocesar datos con caché
@st.cache_data
def load_and_preprocess():
    # Cargar datos
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
    
    # Eliminar columnas
    columns_to_drop = ['budget', 'homepage', 'id', 'original_language', 'production_companies',
                      'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages',
                      'status', 'tagline', 'title']
    df.drop(columns=columns_to_drop, inplace=True)

    # Función para parsear JSON
    def parse_json(data):
        try:
            return ast.literal_eval(data)
        except:
            return []
    
    # Procesar géneros
    df['genres'] = df['genres'].apply(parse_json)
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['genres'] = df['genres'].astype(str)
    
    # Procesar keywords
    df['keywords'] = df['keywords'].apply(parse_json)
    df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['keywords'] = df['keywords'].astype(str)
    
    # Procesar cast
    df['cast'] = df['cast'].apply(parse_json)
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['cast'] = df['cast'].astype(str)
    
    # Procesar director
    def get_director(crew_data):
        crew = parse_json(crew_data)
        for member in crew:
            if member.get('job') == 'Director':
                return member.get('name')
        return None
    df['filmmaker'] = df['crew'].apply(get_director)
    df = df.dropna(subset=['filmmaker'])
    
    # Limpieza adicional
    df = df[(df['vote_average'] != 0)]
    df = df[df['filmmaker'] != '']
    
    # Convertir listas a strings
    df['genres'] = df['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    df['keywords'] = df['keywords'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    df['cast'] = df['cast'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
    
    return df

# Función para entrenar el modelo
@st.cache_data
def train_model(df):
    # Crear listas únicas para características
    genre_list = list(set([genre for sublist in df['genres'].str.split(',') for genre in sublist]))
    keyword_list = list(set([word for sublist in df['keywords'].str.split(',') for word in sublist]))
    cast_list = list(set([actor for sublist in df['cast'].str.split(',') for actor in sublist]))
    
    # Función para codificación binaria
    def binary_encoder(_list, items):
        return [1 if item in items else 0 for item in _list]
    
    # Crear características
    df['genres_bin'] = df['genres'].str.split(',').apply(lambda x: binary_encoder(genre_list, x))
    df['keywords_bin'] = df['keywords'].str.split(',').apply(lambda x: binary_encoder(keyword_list, x))
    df['cast_bin'] = df['cast'].str.split(',').apply(lambda x: binary_encoder(cast_list, x))
    
    # Codificar directores
    mlb = MultiLabelBinarizer()
    director_bin = mlb.fit_transform(df['filmmaker'].apply(lambda x: [x]))
    
    # Escalar características numéricas
    scaler = StandardScaler()
    numerical = scaler.fit_transform(df[['popularity', 'vote_average']])
    
    # Combinar características con pesos
    weights = {
        'genres': 0.33,
        'keywords': 0.27,
        'cast': 0.1,
        'director': 0.18,
        'numerical': 0.12
    }
    
    features = np.concatenate([
        np.array(df['genres_bin'].tolist()) * weights['genres'],
        np.array(df['keywords_bin'].tolist()) * weights['keywords'],
        np.array(df['cast_bin'].tolist()) * weights['cast'],
        director_bin * weights['director'],
        numerical * weights['numerical']
    ], axis=1)
    
    # Entrenar modelo
    nn = NearestNeighbors(n_neighbors=100, metric='cosine', algorithm='brute')
    nn.fit(features)
    
    return nn, df, features  # Modificado para retornar features

# Interfaz de usuario
def main():
    st.title("🎥 Sistema de Recomendación de Películas (Beta 14/02/2025)")
    st.markdown("---")
    
    # Cargar datos y modelo
    df = load_and_preprocess()
    model, df, features = train_model(df)  # Modificado para recibir features
    
    # Sidebar con controles
    with st.sidebar:
        st.header("⚙️ Filtros de Recomendación")
        min_votes = st.slider("Votos Mínimos", 0, 2000, 380)
        min_rating = st.slider("Rating Mínimo", 0.0, 10.0, 5.7)
        num_recs = st.slider("Número de Recomendaciones", 1, 20, 5)
    
    # Búsqueda principal
    # Obtener lista de títulos únicos
    movie_list = df['original_title'].tolist()
    
    # Búsqueda principal con autocompletado
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_input(
            "¿Qué película te gusta?", 
            placeholder="Empieza a escribir...",
            help="Escribe parte del título y verás sugerencias",
            key="movie_input"
        )
    
    # Generar sugerencias
    suggestions = []
    if user_input:
        suggestions = difflib.get_close_matches(
            user_input, 
            movie_list, 
            n=5,
            cutoff=0.4  # Ajustar sensibilidad (0-1)
        )
    
    # Mostrar sugerencias como botones clickables
    if suggestions:
        with col2:
            st.write("🔍 **Sugerencias:**")
            for title in suggestions:
                if st.button(title, key=f"sugg_{title}"):
                    st.session_state.selected_movie = title
    
    # Usar la selección del usuario
    if 'selected_movie' in st.session_state:
        movie_query = st.session_state.selected_movie
    else:
        movie_query = user_input
    
    if st.button("Generar Recomendaciones"):
        if movie_query:
            try:
                idx = df[df['original_title'] == movie_query].index[0]
                
                # Obtener características directamente del array
                distances, indices = model.kneighbors([features[idx]], n_neighbors=100)
                
                recommendations = df.iloc[indices[0]].copy()
                recommendations = recommendations[
                    (recommendations['vote_count'] >= min_votes) &
                    (recommendations['vote_average'] >= min_rating) &
                    (recommendations.index != idx)
                ].head(num_recs)
                
                if not recommendations.empty:
                    st.subheader(f"Recomendaciones basadas en: {movie_query}")
                    for _, row in recommendations.iterrows():
                        with st.expander(row['original_title']):
                            st.markdown(f"**Director:** {row['filmmaker']}")
                            st.markdown(f"**Rating:** {row['vote_average']}/10 ({row['vote_count']} votos)")
                            st.markdown(f"**Popularidad:** {row['popularity']:.1f}")
                            st.markdown(f"**Sinopsis:** {row['overview'][:200]}...")
                else:
                    st.warning("No se encontraron recomendaciones que cumplan los criterios")
                    
            except IndexError:
                st.error("Película no encontrada en la base de datos")
        else:
            st.warning("Por favor ingresa el título de una película")
    
    # Estadísticas del dataset
    st.markdown("---")
    st.subheader("📊 Estadísticas del Dataset")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Películas", len(df))
    
    with col2:
        avg_rating = df['vote_average'].mean()
        st.metric("Rating Promedio", f"{avg_rating:.1f}/10")
    
    with col3:
        total_votes = df['vote_count'].sum()
        st.metric("Votos Totales", f"{total_votes:,}")
    
    # Gráfico de géneros
    st.markdown("### Distribución por Géneros")
    genres = [genre for sublist in df['genres'].str.split(',') for genre in sublist]
    genre_counts = pd.Series(genres).value_counts().head(10)
    st.bar_chart(genre_counts)

if __name__ == "__main__":
    main()