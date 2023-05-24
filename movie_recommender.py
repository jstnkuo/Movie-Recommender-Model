#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import dependencies
import pandas as pd
from tabulate import tabulate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor



# In[2]:


# get movies dataset
movie_path = "Resources/movies.csv"
movie_df = pd.read_csv(movie_path)
print(' ')
print('Display movie_df:')
print(tabulate(movie_df[:5], headers='keys', tablefmt='fancy_grid'))
print(' ')
movie_df.head()


# In[3]:


# Splitting the title column
# Filtering out movies with a year greater than or equal to 2010
movie_df['year'] = movie_df['title'].str.extract(r'\((\d{4})\)')
movie_df['title'] = movie_df['title'].str.replace(r'\s*\(\d{4}\)', '')
movie_df['year'] = movie_df['year'].fillna(0)
movie_df['year'] = movie_df['year'].astype('int')
movie_df = movie_df[movie_df['year']>=2010]
movie_df.head()


# In[6]:


# get a ratings dataset grouped by 'movieId' with the average rating
rating_path = "Resources/ratings_filter.csv"
rating_df = pd.read_csv(rating_path)
# rating_df = rating_df[['movieId', 'rating']].groupby('movieId').mean()
# rating_df = rating_df.reset_index()
rating_df = rating_df[['movieId', 'rating']]
rating_df.head()
print(' ')
print('Display rating_df:')
print(tabulate(rating_df[:5], headers='keys', tablefmt='fancy_grid'))
print(' ')


# In[7]:


# merge the 'movie_df' and 'rating_df' DataFrames together based on the 'movieId' column
new_df = pd.merge(movie_df, rating_df, left_index=False, right_index=False, how='inner', on='movieId')
new_df = new_df.sort_values('movieId')
new_df = new_df.drop(columns=['year'])
new_df.tail()
print(' ')
print('Display new_df:')
print(tabulate(new_df[:5], headers='keys', tablefmt='fancy_grid'))
print(' ')


# In[ ]:





# ### Basic Knowledge of CountVectorizer : https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Split the dataset into training and test sets
train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

# Extract features from genres using CountVectorizer on the training set
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
X_train = vectorizer.fit_transform(train_df['genres'])

# Compute pairwise cosine similarity between genre vectors on the training set
cosine_sim_train = cosine_similarity(X_train)

# Extract features from genres using CountVectorizer on the test set
X_test = vectorizer.transform(test_df['genres'])

# Compute pairwise cosine similarity between genre vectors on the test set
cosine_sim_test = cosine_similarity(X_test)


# In[ ]:





# In[9]:


# reset the index of test_df
test_df = test_df.reset_index()
test_df = test_df.drop(columns=['index'])
test_df.head()


# In[10]:


# Create a function to recommend similar movies based on a given movie from the test set
def recommend_similar_movies(movie_title, n=10):
    # Get the index of the given movie in the test set
    movie_index = test_df[test_df['title'] == movie_title].index
    if len(movie_index) == 0:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return None
    movie_index = movie_index[0]
    
    # Get the cosine similarity scores for the given movie from the test set
    similarity_scores = cosine_sim_test[movie_index]
    
    # Sort the movies based on similarity scores in descending order
    similar_movie_indices = np.argsort(similarity_scores)[::-1][:n]
    similarity_scores = similarity_scores[similar_movie_indices]
    
    # Get the titles, genres, and ratings of top similar movies from the test set
    similar_movies = test_df.loc[similar_movie_indices]['title']
    top_genres = test_df.loc[similar_movie_indices]['genres']
    top_ratings = test_df.loc[similar_movie_indices]['rating']
    
    # Sort the movies by rating in descending order
    sorted_indices = np.argsort(top_ratings)[::-1]
    similar_movies = similar_movies.iloc[sorted_indices]
    top_genres = top_genres.iloc[sorted_indices]
    top_ratings = top_ratings.iloc[sorted_indices]
    
    return similar_movies, similarity_scores, top_genres, top_ratings


# In[11]:


# make a title list from test_df
test_title = test_df['title'].tolist()
print('----------------------------')
print(f'There are {len(test_title)} movies in the test dataset')


# In[12]:


# make a fuction to plot a bar chart showing the ratings of recommended movies.
def plot_similarity_scores(movies, scores):
    plt.figure(figsize=(10, 6))
    plt.bar(movies, scores, color='blue')
    plt.xlabel('Movies')
    plt.ylabel('Ratings')
    plt.xticks(rotation=90)
    plt.title('Scores of Top10 Movie')
    plt.show()


# In[15]:


# That the user could enter a movie number within the range of 0-4016 
# and convert it into the corresponding movie name.
input_movie_order = int(input("choose and type a number (0 to 4016): "))
input_movie = test_title[input_movie_order]
print('----------------------------')
print(input_movie)


# In[16]:


def evaluate_recommendations_by_genres(input_movie):
    # Find the actual genres in the DataFrame based on the movie title
    actual_genres = new_df.loc[new_df['title'] == input_movie, 'genres'].tolist()
    if not actual_genres:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return None
    actual_genres = actual_genres[0].split('|')

    # Get the recommended movies and their genres
    similar_movies, _, top_genres, _ = recommend_similar_movies(input_movie, n=10)

    # Get the genres of the recommended movies
    recommended_genres = top_genres.str.split('|')

    # Compute the accuracy score based on genres
    accuracy = sum(any(genre in actual_genres for genre in movie_genres) for movie_genres in recommended_genres) / len(recommended_genres)

    return accuracy


# In[17]:


actual_genres = new_df.loc[new_df['title'] == 'Appetites']
type(actual_genres['genres'].tolist())


# In[18]:


def show_results():
    similar_movies, similarity_scores, top_genres, top_ratings = recommend_similar_movies(input_movie, n=10)
    if similar_movies is not None:
        print(f"Recommended movies similar to '{input_movie}':")

        for movie, similarity, rating, genres, index in zip(similar_movies, similarity_scores, top_ratings, top_genres, range(1, 11)):
            print(f"Top{index} Movie: {movie}, Similarity: {round(similarity, 7)}, Genres: {genres}")
    score = evaluate_recommendations_by_genres(input_movie)
    
    print('----------------------------')
    print('----------------------------')
    print(f'accuracy score is {score}')
    plot_similarity_scores(similar_movies, top_ratings)


# In[19]:


show_results()


# In[ ]:

genre_feats=vectorizer.transform(new_df['genres'])

X=genre_feats.todense()
y=new_df['rating']



# In[ ]:


rf=RandomForestRegressor()
rf.fit(X,  y)


# In[ ]:


def predict_ratings():
    user_genre_input = input('choose a genre: ')
    user_input_vector=vectorizer.transform([user_genre_input])
    print('----------------------------')
    print(f'the predicted rating for your genre: {rf.predict(user_input_vector)}')


# In[ ]:

predict_ratings()


