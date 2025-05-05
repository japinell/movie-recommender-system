# Movie Recommendation System
# This script uses the framework provided by Bernd Schrooten in Building a Movie Recommendation System published by Packt Publishing
# Code generated using DataCamp AI's Python code generation tool
import pandas as pd
import matplotlib.pyplot as plt

# Load data files
def load_data():
    """
    Load movie metadata and ratings data from CSV files.

    Returns:
    - DataFrame: Movies metadata DataFrame.
    - DataFrame: Ratings DataFrame.
    """
    # Read the movies_metadata file
    movies_metadata = pd.read_csv('./data/movies_metadata.csv')
    
    # Read the ratings file
    ratings = pd.read_csv('./data/ratings.csv')

    return movies_metadata, ratings

# Plot the distribution of the movies metadata
def plot_movies_metadata():
    """
    Plot the distribution of the movies metadata.
    """
    # Visualise the vote_average column
    plt.figure(figsize=(10, 5))
    plt.hist(movies_metadata['vote_average'].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Vote Averages')
    plt.xlabel('Vote Average')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Visualise the vote_count column
    plt.figure(figsize=(10, 5))
    plt.hist(movies_metadata['vote_count'].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Vote Counts')
    plt.xlabel('Vote Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot the distribution of the ratings
def plot_ratings():
    """
    Plot the distribution of the ratings.
    """
    # Visualise the distribution of the rating column
    plt.figure(figsize=(10, 5))
    plt.hist(ratings['rating'].dropna(), bins=10, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Load movie metadata and ratings files
movies_metadata, ratings = load_data()
#plot_movies_metadata()
#plot_ratings()

# Simple recommender based on popularity or high rating
def simple_recommender(criterion='vote_average', top_n=10):
    """
    Generate a simple recommender based on the specified criterion.

    Parameters:
    - criterion (str): Criterion to sort movies by. Either 'vote_average' or 'vote_count'.
    - top_n (int): Number of top movies to recommend.

    Returns:
    - DataFrame: Top N recommended movies based on the specified criterion.
    """
    if criterion not in ['vote_average', 'vote_count']:
        raise ValueError("Criterion must be either 'vote_average' or 'vote_count'")

    # Sort movies by the specified criterion in descending order
    recommended_movies = movies_metadata.sort_values(by=criterion, ascending=False)

    # Select relevant columns to display
    recommended_movies = recommended_movies[['title', 'overview', criterion]]
    
    # Select top n recommended movies
    top_recommended_movies = recommended_movies.head(top_n)

    return top_recommended_movies

# Example usage:
#top_movies_by_average = simple_recommender( criterion='vote_average', top_n=10)
#top_movies_by_count = simple_recommender(criterion='vote_count', top_n=10)

#print("Top Movies by Average Rating:")
#print(top_movies_by_average)

#print("\nTop Movies by Vote Count:")
#print(top_movies_by_count)

# Generate recommendations based on user ratings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Merge movies metadata with ratings
movies_ratings = pd.merge(ratings, movies_metadata, on='movie_id')

# Create a pivot table with users as rows and movies as columns
user_movie_ratings = movies_ratings.pivot_table(index='user_id', columns='title', values='rating')

# Fill NaN values with 0 (assuming unrated movies have a rating of 0)
user_movie_ratings = user_movie_ratings.fillna(0)

# Compute the cosine similarity matrix
movie_similarity = cosine_similarity(user_movie_ratings.T)

# Create a DataFrame for the cosine similarity matrix
movie_sim_df = pd.DataFrame(movie_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

def rating_based_recommender(movie_title, top_n=10):
    # Check if the movie title exists in the matrix
    if movie_title not in user_movie_ratings.columns:
        raise ValueError(f"Movie '{movie_title}' not found in the dataset.")

    # Get the similarity scores for the given movie
    similar_movies = movie_sim_df[movie_title]
    
    # Sort the movies based on the similarity scores
    similar_movies = similar_movies.sort_values(ascending=False)

    # Exclude the input movie itself
    similar_movies = similar_movies.drop(movie_title)
    
    # Get the top N recommendations 
    top_similar_movies = similar_movies.head(top_n)

    # Merge with movie metadata to get additional information
    top_similar_movies_df = top_similar_movies.reset_index().merge(movies_metadata, left_on='title', right_on='title')   
    
    return top_similar_movies_df

# Example usage
#movie_title = "Gladiator"  # Replace with the movie title you want recommendations for
#recommended_movies = rating_based_recommender(movie_title, top_n=10)

# Display the recommended movies    
#print(f"\nTop 10 recommendations for '{movie_title}':")
#print(recommended_movies[['title', 'overview', 'vote_average', 'vote_count']])

# Generate embeddings based on the movie descriptions
from sentence_transformers import SentenceTransformer
#import pandas as pd
from tqdm import tqdm

# Load a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings():   
    # Ensure the 'overview' column is filled with strings
    movies_metadata['overview'] = movies_metadata['overview'].fillna('').astype(str)

    tqdm.pandas(desc="Generating embeddings...")

    # Generate embeddings for the movie overviews
    movies_metadata['overview_embedding'] = movies_metadata['overview'].progress_apply(lambda x: model.encode(x).tolist())
    
generate_embeddings()

# Display the first few rows to verify the embeddings
#print(movies_metadata[['title', 'overview', 'overview_embedding']].head())

# Use embeddings similarity to generate recommendations
#from sklearn.metrics.pairwise import cosine_similarity
#import numpy as np

def embedding_based_recommender(description, top_n=10):
    # Generate embedding for the user input
    description_embedding = model.encode(description)

    tqdm.pandas(desc="Calculating movie similarities...")
    
    # Calculate cosine similarity between user input embedding and all movie embeddings
    movies_metadata['similarity'] = movies_metadata['overview_embedding'].apply(lambda x: cosine_similarity([description_embedding], [x])[0][0] if x is not None else 0)
    
    # Sort movies by similarity in descending order and get the top n movies
    top_movies = movies_metadata.sort_values(by='similarity', ascending=False).head(top_n)
    
    return top_movies[['title', 'overview', 'similarity']]

# Example usage
#user_description = "A story about a young wizard who discovers his magical heritage."
#recommended_movies = embedding_based_recommender(user_description)
#print(recommended_movies)

# All recommender methods combined
def combined_recommender(recommender_type='vote_average', user_input=None, top_n=10):
    """
    Combines four different recommenders into one function.

    Parameters:
    - recommender_type (str): Type of recommender to use. Options are 'vote_average', 'vote_count', 'similarity', or 'embedding'.
    - user_input (str): User description or movie title.
    - top_n (int): Number of top recommendations to return.

    - movie_title (str): Title of the movie for which to find similar movies (only used if recommender_type is 'similar_movies').

    Returns:
    - list of top_n recommendations
    """
    if recommender_type == 'vote_average':
        # Recommender based on vote_average 
        return simple_recommender(criterion='vote_average', top_n=top_n)
    
    elif recommender_type == 'vote_count':
        # Recommender based on vote_count
        return simple_recommender(criterion='vote_count', top_n=top_n)
    
    elif recommender_type == 'similarity':
        # Recommender based on ratings data for a given movie title
        return rating_based_recommender(user_input, top_n=top_n)
    
    elif recommender_type == 'embedding':
        # Recommender based on movie embeddings for a user-generated prompt
        return embedding_based_recommender(user_input, top_n=top_n)
    
    else:
        raise ValueError("Invalid method or missing parameters")

# Example usage:
user_description = "A story about a young wizard who discovers his magical heritage."
top_recommendations = combined_recommender(recommender_type='vote_average', top_n=10)
print(top_recommendations)
# combined_recommender(method='similarity', movie_title='The Matrix', top_n=5)
# combined_recommender(method='embedding', user_input=['The Matrix', 'Inception'], top_n=5)


