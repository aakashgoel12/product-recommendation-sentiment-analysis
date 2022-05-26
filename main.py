################################
## STEP 01: Import Libraries  ##
################################
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

#############################
## STEP 02: Read Data    ####
#############################
# Reading product review data
df_prod_review = pd.read_csv('https://raw.githubusercontent.com/aakashgoel12/blogs/master/input/product_review.csv',\
                      encoding='latin-1')
display(df_prod_review.sample(n=5, random_state=42))

###########################
## STEP 03: Load Model ####
###########################

model =  pickle.load(open('./model/logit_model.pkl', 'rb'))
word_vectorizer = pickle.load(open('./model/word_vectorizer.pkl','rb'))
user_final_rating = pickle.load(open('./model/user_final_rating.pkl','rb'))

##########################################################################
## STEP 04: Get positive review Recommendation only for given user id ####
##########################################################################

def find_top_recommendations(pred_rating_df, userid, topn):
    recommendation = pred_rating_df.loc[userid].sort_values(ascending=False)[0:topn]
    recommendation = pd.DataFrame(recommendation).reset_index().rename(columns={userid:'predicted_ratings'})
    return recommendation

def get_sentiment_product(x):
    ## Get review list for given product
    product_name_review_list = df_prod_review[df_prod_review['prod_name']== x]['Review'].tolist()
    ## Transform review list into DTM (Document/review Term Matrix)
    features= word_vectorizer.transform(product_name_review_list)
    ## Predict sentiment
    return model.predict(features).mean()

def find_top_pos_recommendation(user_final_rating, user_input, df_prod_review, word_vectorizer,\
                                 model, no_recommendation):
    ## 10 is manually coded, need to change 
    ## Generate top recommenddations using user-user based recommendation system w/o using sentiment analysis  
    recommendation_user_user = find_top_recommendations(user_final_rating, user_input, 10)
    recommendation_user_user['userId'] = user_input
    ## filter out recommendations where predicted rating is zero
    recommendation_user_user = recommendation_user_user[recommendation_user_user['predicted_ratings']!=0]
    print("Recommended products for user id:{} without using sentiment".format(user_input))
    display(recommendation_user_user)
    ## Get overall sentiment score for each recommended product
    recommendation_user_user['sentiment_score'] = recommendation_user_user['prod_name'].apply(get_sentiment_product)
    ## Transform scale of sentiment so that it can be manipulated with predicted rating score
    scaler = MinMaxScaler(feature_range=(1, 5))
    scaler.fit(recommendation_user_user[['sentiment_score']])
    recommendation_user_user['sentiment_score'] = scaler.transform(recommendation_user_user[['sentiment_score']])
    ## Get final product ranking score using 1*Predicted rating of recommended product + 2*normalized sentiment score on scale of 1–5 of recommended product 
    recommendation_user_user['product_ranking_score'] =  1*recommendation_user_user['predicted_ratings'] + \
                                                        2*recommendation_user_user['sentiment_score']
    print("Recommended products for user id:{} after using sentiment".format(user_input))
    ## Sort product ranking score in descending order and show only top `no_recommendation`
    display(recommendation_user_user.sort_values(by = ['product_ranking_score'],ascending = False).head(no_recommendation))

user_input = str(input("Enter your user id"))
find_top_pos_recommendation(user_final_rating, user_input, df_prod_review, word_vectorizer,\
                                 model, no_recommendation = 5)