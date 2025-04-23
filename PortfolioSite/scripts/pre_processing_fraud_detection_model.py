# Importing necessary libraries for preprocessing
import pandas as pd
import numpy as np

# Importing encoder to encode the dataset and scaler to normalize
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importing library to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Importing libraries for different models
from sklearn.ensemble import RandomForestClassifier

# Importing libraries to tune the model
from sklearn.model_selection import StratifiedKFold

# Importing libraries for more models to oversample and undersample
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline, make_pipeline

# Importing joblib to save model
import joblib

# Importing library to remove warnings
import warnings
warnings.filterwarnings('ignore')

# Import credit card transaction dataset with fraudulent identifier
data = pd.read_csv(r'C:\Users\Luis Alfredo\Documents\Data Science - Python\Datasets\FraudData.csv')

# Removing unnecessary columns
data_ = data.drop(['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'trans_num', 'dob', 'job'], axis = 1)

# Frequency encoding merchant column
merchant_frequency = data_['merchant'].value_counts()
data_['encoded_merchant'] = data_['merchant'].map(merchant_frequency)

# dropping non encoded columns 
data_ = data_.drop(['merchant'], axis = 1)

# One hot encoding gender and category columns
data_ = pd.get_dummies(data_, columns = ['gender', 'category'], drop_first = True)

# Separating fraud identifitier column from  
x = data_.drop(['is_fraud'], axis = 1)
y = data_['is_fraud']

# Splitting data for modeling
x_train, x_test, y_train, y_test = train_test_split(x, y , stratify = y, test_size = 0.3, random_state = 42)

# Initializing StratifiedKFolds as our form of cross validation
kf = StratifiedKFold(n_splits = 5, shuffle = False)

# Creating pipeline that applies random oversampling to the dataset followed by a random forest classifier
random_oversample_pipeline = make_pipeline(RandomOverSampler(random_state = 42), RandomForestClassifier(n_estimators = 100, random_state = 1))

# Fit the pipeline on training data
random_oversample_pipeline.fit(x_train, y_train)

# Save the trained pipeline
joblib.dump(random_oversample_pipeline, "C:\Users\Luis Alfredo\Documents\GitHub\portfolio_site\PortfolioSite\staticfiles\ml_models\fraud_detection_model.joblib")