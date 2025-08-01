import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if not poster_path:
        return "https://asset.cloudinary.com/dabemeznd/1075c7a4f3d7d029ea61054b237fd3e6"
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"

def recommend(movie_title, top_n=5):
    movie_row = movies[movies['title'] == movie_title]
    if movie_row.empty:
        raise ValueError(f"Movie '{movie_title}' not found.")

    query_desc = movie_row['description'].values[0]
    if not isinstance(query_desc, str) or not query_desc.strip():
        raise ValueError(f"Invalid description for '{movie_title}'.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    query_vec = model.encode([query_desc], convert_to_numpy=True, batch_size=1)
    query_embedding = np.array(query_vec, dtype=np.float32)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    if not query_embedding.flags.c_contiguous:
        query_embedding = np.ascontiguousarray(query_embedding)

    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_n + 1)
    result_indices = indices[0][1:top_n + 1]
    recommended_movie_titles = []
    recommended_movie_posters = []
    for i in result_indices:
        movie_id = movies.iloc[i]['id']
        recommended_movie_titles.append(movies.iloc[i]['title'])
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie_titles, recommended_movie_posters

st.markdown("""
<style>
.movie-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    max-height: 600px;
}
.movie-title {
    width: 120px;
    min-height: 60px;
    text-align: center;
    margin-bottom: 10px;
}
.movie-poster {
    width: 120px !important;
    height: 180px !important;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)

st.header('Movies Recommendation System')

@st.cache_data
def load_data():
    return pd.read_csv('data/TMDB_movie_dataset_v11_cleaned.csv', usecols=['id', 'title', 'description'])

@st.cache_data
def get_movie_list():
    return movies['title'].dropna().head(50).tolist()

movies = load_data()

@st.cache_resource
def load_model_and_index():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model.to(device)
    index = faiss.read_index("data/movie_faiss.index")
    return model, index

model, index = load_model_and_index()

if 'movie_input' not in st.session_state:
    st.session_state.movie_input = ""
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

movie_list = get_movie_list()
movie_input = st.text_input("Enter a movie title", value=st.session_state.movie_input, key='movie_input')
selected_movie = st.selectbox("Or select a movie from the dropdown", [""] + movie_list, index=0, key='movie_select')

final_movie = movie_input if movie_input.strip() else selected_movie if selected_movie else None

if st.button('Recommend Movies') and final_movie:
    with st.spinner('Generating recommendations...'):
        recommended_movie_titles, recommended_movie_posters = recommend(final_movie)
        cols = st.columns(5)
        for col, title, poster in zip(cols, recommended_movie_titles, recommended_movie_posters):
            with col:
                with st.container():
                    st.markdown(f'<div class="movie-container"><div class="movie-title">{title}</div></div>', unsafe_allow_html=True)
                    if poster:
                        st.image(poster, use_container_width=False, clamp=True, output_format='JPEG')
                    else:
                        st.text("No poster available")
