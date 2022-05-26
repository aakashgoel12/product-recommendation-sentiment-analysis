################################
## STEP 01: Import Libraries  ##
################################
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from imblearn import over_sampling
from IPython.display import display

#############################
## STEP 02: Read Data    ####
#############################
# Reading product review sentiment file
df_prod_review = pd.read_csv('https://raw.githubusercontent.com/aakashgoel12/blogs/master/input/product_review_sentiment.csv',\
                      encoding='latin-1')
display(df_prod_review.sample(n=5, random_state=42))

#################################
## STEP 03: Data Preparation ####
#################################
x=df_prod_review['Review']
y=df_prod_review['user_sentiment']
print("Checking distribution of +ve and -ve review sentiment: \n{}".format(y.value_counts(normalize=True)))
# Split the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

#As we saw above that data is imbalanced, balance training data using over sampling

ros = over_sampling.RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))
print("Checking distribution of +ve and -ve review sentiment after oversampling: \n{}".format(y_train.value_counts(normalize=True)))
#convert into list of string
X_train = X_train['Review'].tolist()


################################################################
## STEP 04: Feature Engineering (Convert text into numbers) ####
################################################################
word_vectorizer = TfidfVectorizer(strip_accents='unicode', token_pattern=r'\w{1,}',\
                                ngram_range=(1, 3), stop_words='english', sublinear_tf=True,\
                                 max_df = 0.80, min_df = 0.01)

# Fiting it on Train
word_vectorizer.fit(X_train)
# transforming the train and test datasets
X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())
# print(list(word_vectorizer.get_feature_names()))

###############################################
## STEP 05: ML Model (Logistic Regression) ####
###############################################

def evaluate_model(y_pred,y_actual):
    print(classification_report(y_true = y_actual, y_pred = y_pred))
    #confusion matrix
    cm = confusion_matrix(y_true = y_actual, y_pred = y_pred)
    TN = cm[0, 0] 
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    #Calculating the Sensitivity
    sensitivity = round(TP/float(FN + TP),2)
    print("sensitivity: {}".format(sensitivity))
    #Calculating the Specificity
    specificity = round(TN / float(TN + FP),2)
    print("specificity: {}".format(specificity))

#4.1 Model Training
logit = LogisticRegression()
logit.fit(X_train_transformed,y_train)
#4.2 Prediction on Train Data
y_pred_train= logit.predict(X_train_transformed)
#4.3 Prediction on Test Data
y_pred_test = logit.predict(X_test_transformed)
#4.4 Evaluation on Train
print("Evaluation on Train dataset ..")
evaluate_model(y_pred = y_pred_train, y_actual = y_train)
print("Evaluation on Test dataset ..")
#4.5 Evaluation on Test
evaluate_model(y_pred = y_pred_test, y_actual = y_test)

############################
## STEP 06: Save Model  ####
############################
pickle.dump(logit,open('./model/logit_model.pkl', 'wb'))
pickle.dump(word_vectorizer,open('./model/word_vectorizer.pkl','wb'))