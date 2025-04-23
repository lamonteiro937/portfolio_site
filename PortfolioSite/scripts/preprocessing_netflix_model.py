# Importing necessary libraries for preprocessing
import pandas as pd
import numpy as np

# Importing to load datasets, parse file containing ratings, and calculate the accuracy of models
from surprise.reader import Reader
from surprise.dataset import Dataset

# Importing KNN for similarity based recommendations, train_test_split to split dataset into training and testing sets, and joblib to save models
from surprise.model_selection import train_test_split
from surprise.prediction_algorithms.knns import KNNBasic
import joblib

# Importing library to remove warnings
import warnings
warnings.filterwarnings('ignore')

# Import Netflix datasets for movies and user ratings on movies
movies_ = pd.read_csv(r'C:\Users\Luis Alfredo\Documents\Data Science - Python\Datasets\movies.csv')
ratings = pd.read_csv(r'C:\Users\Luis Alfredo\Documents\Data Science - Python\Datasets\Netflix_User_Ratings.csv')

# Merging the movies dataset to user ratings dataset
data = pd.merge(ratings, movies_, on = 'MovieId', how = 'left')

# Choosing 23,000 random integers from 0 to length of unique users
random_ints = list(np.random.choice(np.arange(0,len(data['CustId'].unique())), size = 23000, replace = False))

# Using the random 23,000 integers to pick users from a list using the integers as indices
random_users = data['CustId'].unique()[[random_ints]]

# Using the random users list to select the users within the dataframe
data = data[(data.CustId.isin(random_users))]

# Putting the 23,000 random users into a list 
users = data.CustId

# Creating a dictionary for users and counting the number of ratings made by each user 
rating_counts = dict()

# If user has been accounted for then we add 1 to the rating count, otherwise we start the user rating count at 1, CustId is the dictionary key 
for user in users:
    if user in rating_counts:
        rating_counts[user] += 1

    else:
        rating_counts[user] = 1

# Number of ratings made by a user has a cutoff of 100
ratings_cutoff = 100

# Creating list of users to be removed
remove_users = []

# If user does not have at least 100 rating counts, it will be added to list of users to be removed
for user, num_ratings in rating_counts.items():
    if num_ratings < ratings_cutoff:
        remove_users.append(user)

# Removing users from list that do not have at least 100 ratings count
data = data.loc[~data.CustId.isin(remove_users)]

# Putting the movies into a list 
movies = data.MovieId

# Creating a dictionary for movies and counting the number of ratings made for each movie
ratings_count = dict()

# If movie has been accounted for then we add 1 to the rating count, otherwise we start the movie rating count at 1, MovieId is the dictionary key 
for movie in movies:
    if movie in ratings_count:
        ratings_count[movie] += 1

    else:
        ratings_count[movie] = 1

# Number of ratings a movie has a cutoff of 500
rating_cutoff = 500

# Creating list of movies to be removed
remove_movies = []

# If movie does not have at least 500 rating counts, it will be added to list of movies to be removed
for movie, num_ratings in ratings_count.items():
    if num_ratings < rating_cutoff:
        remove_movies.append(movie)

# Removing movies from list that do not have at least 500 ratings count
data = data.loc[~data.MovieId.isin(remove_movies)]

# Creating list of more users to be removed
remove_users_ = []

# If user does not use a range of rating, they will be removed
for user__ in data['CustId'].unique():
    if len(data[data['CustId'] == user__]['Rating'].unique()) < 4:
        remove_users_.append(user__)

data = data.loc[~data.CustId.isin(remove_users_)]

# Copying dataframe
df = data.copy()

# Importing encoder to ease the processing of the dataset
from sklearn.preprocessing import LabelEncoder

# Encoding columns CustId and MovieId
le = LabelEncoder()

df['CustId'] = le.fit_transform(df['CustId'])
df['MovieId'] = le.fit_transform(df['MovieId'])

# Instantiating Reader scale with expected rating scale
reader = Reader(rating_scale = (1, 5))

# Loading dataset
_data = Dataset.load_from_df(df[['CustId', 'MovieId', 'Rating']], reader)

# Splitting data into training and testing sets
trainset, testset = train_test_split(_data, test_size = 0.4, random_state = 42)

# User-Based collaborative filtering recommendation system model parameters
sim_options = {'name':  'msd', 'user_based': True}

# KNN for user similarity-based recommendations  
user_based_model = KNNBasic(sim_options = sim_options, k = 30, min_k = 3, verbose = True)

# Training algorithm with training set
user_based_model.fit(trainset)

# Save Model
joblib.dump(user_based_model, r"C:\Users\Luis Alfredo\Documents\GitHub\portfolio_site\PortfolioSite\staticfiles\ml_models\netflix_user_base_model.joblib")
# Item-Based collaborative filtering recommendation system model parameters
sim_options = {'name': 'msd', 'user_based': False}

# KNN for item similarity-based recommendations
item_based_model = KNNBasic(sim_options = sim_options, k = 20, min_k = 3, verbose = False)

# Training algorithm with training set
item_based_model.fit(trainset)

# Save Model
joblib.dump(item_based_model, r"C:\Users\Luis Alfredo\Documents\GitHub\portfolio_site\PortfolioSite\staticfiles\ml_models\netflix_item_base_model.joblib")