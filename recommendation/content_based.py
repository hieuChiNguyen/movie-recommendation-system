import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def compute_content_based_features(movies: pd.DataFrame) -> tuple:
    """
    Compute TF-IDF matrix, movie indices, and cosine similarity matrix for content-based
    recommendation using movie descriptions.

    Args:
        movies (pd.DataFrame): DataFrame containing movie data.

    Returns:
        tuple: (tfidf_matrix, indices, cosine_sim)
            - tfidf_matrix: TF-IDF matrix of movie descriptions.
            - indices: Series mapping movie titles to DataFrame indices.
            - cosine_sim: Cosine similarity matrix for all movies.
    """
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Compute TF-IDF matrix
    print("Computing TF-IDF matrix...")
    tfidf_matrix = tfidf.fit_transform(movies['description'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Create index mapping for movie titles
    indices = pd.Series(movies.index, index=movies['title'])
    print(f"Indices mapping: {indices.head()}")

    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return tfidf_matrix, tfidf, indices, cosine_sim


def get_recommendations(movie_title: str, cosine_sim: np.ndarray, indices: pd.Series,
                        movies: pd.DataFrame, top_n: int = 5) -> pd.Series:
    """
    Get top-N movie recommendations based on cosine similarity for a given movie title.

    Args:
        movie_title (str): Title of the movie to find recommendations for.
        cosine_sim (np.ndarray): Cosine similarity matrix.
        indices (pd.Series): Series mapping movie titles to DataFrame indices.
        movies (pd.DataFrame): DataFrame containing movie data.
        top_n (int): Number of recommendations to return (default: 5).

    Returns:
        pd.Series: Titles of the top-N recommended movies.
    """
    # Get movie index
    idx = indices[movie_title]

    # Get similarity scores for the movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Select top-N similar movies (excluding the input movie)
    sim_scores = sim_scores[1:top_n + 1]
    print(f"Top {top_n} recommendations for '{movie_title}': {sim_scores}")

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    print(f"Movie indices for recommendations: {movie_indices}")

    # Return recommended movie titles
    return movies['title'].iloc[movie_indices]


def main():
    # File path for the movie dataset
    file_path = 'data/TMDB_movie_dataset_v11_cleaned.csv'

    # Load movie data
    movies = pd.read_csv(file_path)

    # Compute content-based features
    tfidf_matrix, tfidf, indices, cosine_sim = compute_content_based_features(movies)

    # Example recommendation
    movie_title = "The Avengers"
    recommendations = get_recommendations(movie_title, cosine_sim, indices, movies, top_n=5)

    # Display results
    print(f"Recommendations for '{movie_title}':")
    print(recommendations)

    # Display TF-IDF matrix preview
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    print("\nTF-IDF Matrix Preview:")
    print(f"Shape: {tfidf_df.shape}")
    print(tfidf_df.head())


if __name__ == "__main__":
    main()