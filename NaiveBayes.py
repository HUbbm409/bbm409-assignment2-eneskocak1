
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np





def naive_bayes(real_vectorizer,real_word_matrix,real_frequency_dict,fake_vectorizer,fake_word_matrix,fake_frequency_dict,test_line_list,BoW_gram,stopWords=None):
    #####################################################
    # GETTING UNIQ WORDS FOR CALCULATE TOTAL UNIQ WORDS #
    #####################################################
    realwords_set =real_vectorizer.get_feature_names()
    fakewords_set =fake_vectorizer.get_feature_names()
    ######################################################
    #                   NAIVE BAYES PART                 #
    ######################################################
    real_words_count = real_word_matrix.sum() # WORDS COUNT IN REAL NEWS
    fake_words_count = fake_word_matrix.sum() # WORDS COUNT IN FAKE NEWS
    p_real=real_words_count/(real_words_count+fake_words_count)# REAL NEW LİKELİHOOD
    p_fake=fake_words_count/(real_words_count+fake_words_count)# FAKE NEW LİKELİHOOD
    

    uniq_words = set(realwords_set + fakewords_set)
    uniq_words_count = len(uniq_words) # UNIQ WORDS COUNT IN FAKE AND REAL AND TEST NEWS
    # computing bayes theorem with laplace smoothing #
    test_classifier_result=[]
    for line in test_line_list:
        test_vectorizer = CountVectorizer(ngram_range=(BoW_gram,BoW_gram),stop_words=stopWords,analyzer='word')
        line_list= []
        line_list.append(line)
        test_word_matrix = test_vectorizer.fit_transform(line_list)
                
        probabilty_real=0
        probabilty_fake=0
        

        # calculating conditional probabiltys here and if word not in uniq set applying smoothing.
        for item,value in test_vectorizer.vocabulary_.items():
            probabilty_real += test_word_matrix.toarray()[0,value]*np.log10((real_frequency_dict.get(item,0)+1)/(real_words_count + uniq_words_count)) #sum of probabilties real
            probabilty_fake += test_word_matrix.toarray()[0,value]*np.log10((fake_frequency_dict.get(item,0)+1)/(fake_words_count + uniq_words_count)) #sum of probabilties fake
                   
        probabilty_real= probabilty_real + np.log10(p_real)# probabilty of news is real
        probabilty_fake= probabilty_fake + np.log10(p_fake)# probabilty of news is fake
        if probabilty_real > probabilty_fake:
    
            test_classifier_result.append("real")
        elif probabilty_real < probabilty_fake:
      
            test_classifier_result.append("fake")
        elif p_real>=p_fake:
            test_classifier_result.append("real")
        else:
            test_classifier_result.append("fake")        
    return test_classifier_result
    
    

