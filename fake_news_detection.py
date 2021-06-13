#Importing Libraries
from nltk import stem
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, strip_tags
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download("stopwords")
# print(stopwords.words("english")) 

#Data Pre-Processing
news_dataset= pd.read_csv(r"C:\Users\Paarth Bathla\Documents\Financial Management System_Alpha Hacks\train_fakenews.csv")
print(news_dataset.shape)
print(news_dataset.head())

#missing values
print(news_dataset.isnull().sum())

news_dataset = news_dataset.fillna("")

#Merging Title and Author Columns
news_dataset["content"]=news_dataset['author'] + news_dataset["title"]
print(news_dataset["content"])

#Seperating the Data Label
x = news_dataset.drop(columns="label",axis=1)
y=news_dataset["label"]
print(x)
print(y)

#Stemming Procedure
port_stem= PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset["content"]=news_dataset["content"].apply(stemming)
print(news_dataset['content'])

x= news_dataset['content'].values
y=news_dataset['label'].values

print(x)
print(y)

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
X = vectorizer.transform(x)
print(X)


#Train Test Split
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, stratify=y,random_state=2) 



#Training Logistic Regression Model
model= LogisticRegression()
model.fit(x_train, y_train)

#Evaluation and Accuracy Score

#Training Data Scores
x_train_predict= model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict,y_train)
print(f"Accuracy Score of Training Data => {training_data_accuracy}")

#Test Data Scores
x_test_predict= model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict,y_test)
print(f"Accuracy Score of Test Data => {test_data_accuracy}")

#Predictive Sytem
"""
In x_new=x_test[]
use 
"""
print(y_test[3])

x_new=x_test[3]

prediction= model.predict(x_new)
print(prediction)

if (prediction[0]==0):
    print("The News is Real")
else:
    print("The News is Fake")

print(y_test)
