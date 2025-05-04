# Movie Recommendation System
# This script uses the framework provided by Bernd Schrooten in Building a Movie Recommendation System published by Packt Publishing
# Code generated using DataCamp AI's Python code generation tool
import pandas as pd
import matplotlib.pyplot as plt

# Read the movies_metadata file
movies_metadata = pd.read_csv('./data/movies_metadata.csv')
movies_metadata.head

# Count how many unique movies there are
unique_movies_count = movies_metadata['movie_id'].nunique()
unique_movies_count

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

# Read the ratings file
ratings = pd.read_csv('./data/ratings.csv')
ratings.head()

# Count how many unique users have rated how many unique movies
unique_users_count = ratings['user_id'].nunique()
unique_rated_movies_count = ratings['movie_id'].nunique()

# Display the unique counts of movies and users
print(f"Unique Movies Count: {unique_movies_count}")
print(f"Unique Users Count: {unique_users_count}")
print(f"Unique Rated Movies Count: {unique_rated_movies_count}")

# Visualise the distribution of the rating column
plt.figure(figsize=(10, 5))
plt.hist(ratings['rating'].dropna(), bins=10, edgecolor='k', alpha=0.7)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()