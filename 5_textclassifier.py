# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 21:43:31 2023

@author: Aditya Vikram Sharma
"""
#Text Classifier

#1. Importing Libraries
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords') #for any updated stopwords

#2. Importing the Dataset (Only First Time)
'''reviews=load_files('txt_sentoken/') #dataset with classes'''
#Class 0-negetive Class 1-positive
#below x contains the data and y contains the classes
'''x,y=reviews.data,reviews.target'''


#3. Storing as Pickle Files(Byte type files)
#Used for persisting the dataset
'''
with open('1_x.pickle','wb') as f:
    pickle.dump(x,f)

with open('2_y.pickle','wb') as f:
    pickle.dump(y,f)
'''    

#4. Unpickling the dataset
with open('1_x.pickle','rb') as f:
    x=pickle.load(f)
    
with open('2_y.pickle','rb') as f:
    y=pickle.load(f)


#5. Preprocessing the data, Creating the corpus
corpus=[]

for i in range(len(x)):
    review=re.sub(r'\W',' ',str(x[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+',' ',review)
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)

#6. Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
#creating an object of CountVectorizer class
cv=CountVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))

bow=cv.fit_transform(corpus).toarray()

#7. Transforming the Bag of Words Model to a Tfidf Model
from sklearn.feature_extraction.text import TfidfTransformer
#Creating an object of TfidfTransformer
transformer=TfidfTransformer()

tfidf=transformer.fit_transform(bow).toarray()


#8. Splitting the Dataset into Training and Testing Datasets
from sklearn.model_selection import train_test_split

text_train,text_test,sent_train,sent_test=train_test_split(tfidf,y,test_size=0.2,random_state=0) 
#text-->tfidf values & sent-->sentiment/class


#9. Training the Logistic Regression classifier
from sklearn.linear_model import LogisticRegression
#creating object of LogisticRegression class
classifier=LogisticRegression()
classifier.fit(text_train,sent_train)


#10. Testing our model and checking accuracy of predictions
sent_pred=classifier.predict(text_test) #sentiment predictions

#Creating a confusion matrix and a Heatmap
from sklearn.metrics import confusion_matrix #a function
#pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns

#create confusion matrix
cm=confusion_matrix(sent_test, sent_pred)

#create heatmap
fig, ax = plt.subplots()

sns.heatmap(cm,cmap="YlGnBu",annot=True,fmt="d",cbar_kws={"label":"Scale"},xticklabels=[0,1],yticklabels=[0,1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

#saving plot as high quality image
image_format = 'png' # e.g .png, .svg, etc.
image_name = '6_heatmap.png'

fig.savefig(image_name, format=image_format, dpi=1200)

#Calculating the accuracy of our model
accuracy=((cm[0][0]+cm[1][1])/len(sent_test))*100
print("Accuracy=",accuracy,"%\n")

#We can also calculate the accuracy using a function
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(sent_test, sent_pred)
print("Accuracy=",accuracy)


#11. Saving the vectorizer and the trained classifier as pickle files

from sklearn.feature_extraction.text import TfidfVectorizer

#instead of creating a bow model then converting to tfid model here we are directly creating a tfid model to store as pickle file
tf=TfidfVectorizer(max_features=2000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))

tfidf=tf.fit_transform(corpus).toarray()

with open('3_tfidfmodel.pickle','wb') as f:
    pickle.dump(tf,f)

with open('4_classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    

# 12. Plotting the logistic Regression Curve
#yet to be done