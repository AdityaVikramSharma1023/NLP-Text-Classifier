# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:46:18 2023

@author: Aditya Vikram Sharma
"""
#Using our Model that we created in the previous step

import pickle
#Importing our classifier and vectorizer by unpickling them

with open('3_tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f) #vectorizer
with open('4_classifier.pickle','rb') as f:
    clf=pickle.load(f) #classifier

#Now predicting the sentiment of a sample text

txt=[input("Enter sample text:\n")]
sample=tfidf.transform(txt).toarray()
prediction=clf.predict(sample)
if prediction[0]==1:
    print("\nPositive Text[1]")
else:
    print("\nNegetive text[0]")