
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np


def Vectorizer(realstring,fakestring,BoW_gram,stopWords=None):
    
    ##############################################
    # CREATING LINE-WORD MATRIX WITH FREQUENCY   #
    ##############################################
    real_vectorizer = CountVectorizer(ngram_range=(2,BoW_gram),stop_words=stopWords,analyzer='word')
    real_word_matrix = real_vectorizer.fit_transform(realstring)
    
    fake_vectorizer = CountVectorizer(ngram_range=(2,BoW_gram),stop_words=stopWords,analyzer='word')
    fake_word_matrix = fake_vectorizer.fit_transform(fakestring)
    
    real_vocabulary = real_vectorizer.vocabulary_
    fake_vocabulary = fake_vectorizer.vocabulary_
    
    return real_vectorizer,real_word_matrix,real_vocabulary,fake_vectorizer,fake_word_matrix,fake_vocabulary
############################################################################################################ 
###########################################################################################################
def tfidf_Vectorizer(realstring,fakestring,BoW_gram,stopWords=None):
    
    ##############################################
    # CREATING LINE-WORD MATRIX WITH FREQUENCY   #
    ##############################################
    real_vectorizer = TfidfVectorizer(ngram_range=(2,BoW_gram),stop_words=stopWords,analyzer='word')
    real_word_matrix = real_vectorizer.fit_transform(realstring)
    
    fake_vectorizer = TfidfVectorizer(ngram_range=(2,BoW_gram),stop_words=stopWords,analyzer='word')
    fake_word_matrix = fake_vectorizer.fit_transform(fakestring)
    
    real_vocabulary = real_vectorizer.vocabulary_
    fake_vocabulary = fake_vectorizer.vocabulary_
    
   

    
        
    return real_vectorizer,real_word_matrix,real_vocabulary,fake_vectorizer,fake_word_matrix,fake_vocabulary
############################################################################################################ 