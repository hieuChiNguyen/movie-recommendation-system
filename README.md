### Movie Recommendation System
**1. Content based:** a recommendation technique that relies on the features of the items (such as movies, books, products) to suggest similar items that match a user's preferences.

- Props and cons of content based recommendation system:
  - Pros:
    - Personalized for each user
    - Recommendations are explainable (based on content similarity)

  - Cons:
    - Struggles if item descriptions are poor
    - Cannot discover new trends, lacks diversity


I built a content based recommendation system based on description include some attributes of the movies:
- Overview
- Tagline
- Keywords
- Genres
- Production Companies
- Production Countries


**2. Collaborative filtering:** a recommendation technique that relies on user interactions (such as ratings, clicks, purchases) to suggest items based on the behavior of similar users.
It means user A likes movies X, user B also likes movies X, so user A maybe like movies that user B likes and vice versa.
- Processing


**3. Hybrid:** a combination of content-based and collaborative filtering methods to provide more accurate recommendations.


### Dataset:
- https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies


### References:
- https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system/notebook?select=tmdb_5000_credits.csv

