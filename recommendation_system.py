################################
## STEP 01: Import Libraries  ##
################################
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split 
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from IPython.display import display

#############################
## STEP 02: Read Data    ####
#############################
# Reading ratings file
ratings = pd.read_csv('https://raw.githubusercontent.com/aakashgoel12/blogs/master/input/product_ratings_final.csv',\
                      encoding='latin-1')
# ratings.reset_index(drop=True, inplace=True)
display(ratings.sample(n=5, random_state=42))

#################################
## STEP 03: Data Preparation ####
#################################

def apply_pivot(df,fillby = None):
    if fillby is not None:
        return df.pivot_table(index='userId', columns='prod_name',values='rating').fillna(fillby)
    return df.pivot_table(index='userId', columns='prod_name',values='rating')


#3.1 Dividing the dataset into train and test
train, test = train_test_split(ratings, test_size=0.30, random_state=42)
test = test[test.userId.isin(train.userId)]
#3.2 Apply pivot operation and fillna used to replace NaN values with 0 i.e. where user didn't made any rating
df_train_pivot = apply_pivot(df = train, fillby = 0)
df_test_pivot = apply_pivot(df = test, fillby = 0)
#3.3 dummy dataset (train and test)
## Train
dummy_train = train.copy()
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x>=1 else 1)
dummy_train = apply_pivot(df = dummy_train, fillby = 1)
## Test
dummy_test = test.copy()
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = apply_pivot(df = dummy_test, fillby = 0)

#####################################
## STEP 04: User-User Similarity ####
#####################################

# to calculate mean, use only ratings given by user instead of fillna by 0 as it increase denominator in mean
mean = np.nanmean(apply_pivot(df = train), axis = 1)
df_train_subtracted = (apply_pivot(df = train).T-mean).T
# Make rating=0 where user hasn't given any rating
df_train_subtracted.fillna(0, inplace = True)
# Creating the User Similarity Matrix using pairwise_distance function. shape of user_correlation is userXuser i.e. 18025X18025
user_correlation = 1 - pairwise_distances(df_train_subtracted, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
# user_correlation[user_correlation<0] = 0
# Convert the user_correlation matrix into dataframe
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df['userId'] = df_train_subtracted.index
user_correlation_df.set_index('userId',inplace=True)
user_correlation_df.columns = df_train_subtracted.index.tolist()

###########################################
## STEP 05: Predict Rating (User-User) ####
###########################################
# Rating predicted by the user (for rated & non rated product both) is the weighted sum of correlation with the product rating (as present in the rating dataset). 
user_predicted_ratings = np.dot(user_correlation, df_train_pivot)

# To find only product not rated by the user, ignore the product rated by the user by making it zero. 
user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

# scaler = MinMaxScaler(feature_range=(1, 5))
# scaler.fit(user_final_rating)
# user_final_rating = scaler.transform(user_final_rating)

################################################################
## STEP 06: Find Top N recommendation for User (User-User) #####
################################################################

def find_top_recommendations(pred_rating_df, userid, topn):
    recommendation = pred_rating_df.loc[userid].sort_values(ascending=False)[0:topn]
    recommendation = pd.DataFrame(recommendation).reset_index().rename(columns={userid:'predicted_ratings'})
    return recommendation

user_input = str(input("Enter your user id"))
recommendation_user_user = find_top_recommendations(user_final_rating, user_input, 5)
recommendation_user_user['userId'] = user_input

print("Recommended products for user id:{} as below".format(user_input))
display(recommendation_user_user)
print("Earlier rated products by user id:{} as below".format(user_input))
display(train[train['userId']==user_input].sort_values(['rating'],ascending=False))

################################################
## STEP 07: Evaluation (User-User) on test #####
################################################s

#Filter user correlation only for user which is in test, test is subset/equal of train in terms of userId

user_correlation_test_df = user_correlation_df[user_correlation_df.index.isin(test.userId)]
user_correlation_test_df = user_correlation_test_df[list(set(test.userId))]
# user_correlation_test_df[user_correlation_test_df<0]=0

#Get test user predicted rating
test_user_predicted_ratings = np.dot(user_correlation_test_df, df_test_pivot)
test_user_predicted_ratings = np.multiply(test_user_predicted_ratings,dummy_test)
#Get NaN where user never rated as it shouldn't contribute in calculating RMSE
test_user_predicted_ratings = test_user_predicted_ratings[test_user_predicted_ratings>0]
scaler = MinMaxScaler(feature_range=(1, 5))
scaler.fit(test_user_predicted_ratings)
test_user_predicted_ratings = scaler.transform(test_user_predicted_ratings)

total_non_nan = np.count_nonzero(~np.isnan(test_user_predicted_ratings))
rmse = (np.sum(np.sum((apply_pivot(df = test) - test_user_predicted_ratings)**2))/total_non_nan)**0.5
print(rmse)

############################
## STEP 08: Save Model  ####
############################
pickle.dump(user_final_rating,open('./model/user_final_rating.pkl','wb'))