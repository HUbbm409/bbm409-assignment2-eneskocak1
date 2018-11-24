from NaiveBayes import naive_bayes
from Results import Accuracy
from Optimizations import list_tokenizer,dedect_specific_words
from BagOfWords import tfidf_Vectorizer,Vectorizer
from readfile import readfile
import pandas as pd

"""change diection if you want try diffrent datas"""
#########################################################################################################
real="clean_real-Train.txt"
fake="clean_fake-Train.txt"
test="test.csv"
##########################################################################################################

def All_Results(number):
    ngramlist=[]
    stopwordlist=[]
    stemlist=[]
    tfidflist=[]
    correctlist=[]
    accuracylist=[]
    for n in range(1,number+1):
        a,b,c,d,e,f=main(n,stopWords=None,Stemming=False,tfidf=False)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords=None,Stemming=False,tfidf=True)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords=None,Stemming=True,tfidf=False)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords=None,Stemming=True,tfidf=True)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords="english",Stemming=False,tfidf=False)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords="english",Stemming=False,tfidf=True)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords="english",Stemming=True,tfidf=False)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)
        a,b,c,d,e,f=main(n,stopWords="english",Stemming=True,tfidf=True)
        ngramlist.append(a),stopwordlist.append(b),stemlist.append(c),tfidflist.append(d),correctlist.append(e),accuracylist.append(f)

    t = pd.DataFrame({"N_gram":ngramlist, 'Stop words':stopwordlist,"Stem":stemlist,"TF-IDF":tfidflist,"Correct classified":correctlist,"Accuracy":accuracylist},) # Create a new table with 5 rows and 3 columns
    print(t)
    
    


def main(BoW_gram,stopWords=None,tfidf=False,Stemming=False,PrintCommand=None):
                            #READING DATAS
    #########################################################################################
    real_line_list,fake_line_list,test_line_list,truelist,testfile = readfile(real,fake,test)
    list_tokenizer(real_line_list,fake_line_list,test_line_list,BoW_gram,Stemming)
                            #CONVERT VECTORIZER FORM LINES
    #########################################################################################
    if tfidf==True:
        real_vectorizer,real_word_matrix,real_vocabulary,fake_vectorizer,fake_word_matrix,fake_vocabulary = tfidf_Vectorizer(real_line_list,fake_line_list,BoW_gram,stopWords)
    else:
        real_vectorizer,real_word_matrix,real_vocabulary,fake_vectorizer,fake_word_matrix,fake_vocabulary = Vectorizer(real_line_list,fake_line_list,BoW_gram,stopWords)
                            #DEDECT SPECIFIC WORDS FOR CLASSIFIER NEWS
    #########################################################################################
    real_frequency_dict,fake_frequency_dict=dedect_specific_words(real_word_matrix,real_vocabulary,fake_word_matrix,fake_vocabulary,specific=PrintCommand)
                            #NAIVE BAYES ALGORITHM
    #########################################################################################
    classifier_list=naive_bayes(real_vectorizer,real_word_matrix,real_frequency_dict,fake_vectorizer,fake_word_matrix,fake_frequency_dict,test_line_list,BoW_gram,stopWords)
    #############################################################################################
    ##                      CALCULATING ACCURACY IN HERE 
    accuracy,correctedclassified = Accuracy(truelist,classifier_list,PrintCommand=PrintCommand)
    """here for kaggle output"""
    #testfile["Category"]=classifier_list
    #testfile.to_csv('out.csv ',encoding='utf-8', index=False)
    #print(testfile)
    if PrintCommand=="General Results":
        print("Steamming :",Stemming)
        print("TF-IDF :",tfidf)
        print("The occurrences of words :",BoW_gram)
        print("Stopwords:",stopWords)
        print("Accuracy: ",accuracy)
        print("####################################")
    if PrintCommand==None:
        return BoW_gram,stopWords,Stemming,tfidf,correctedclassified,accuracy


""""OPEN "All_Result" FUNCTION FOR RESULT ALL STUATIONS OR open "main" function for RESULT SPECIFIC SELECTIONS
    main functions first paramater for unigram(1) or bigram(2)
    PrintCommand gives diffrent 4 parameters:
        1. "General Results" : returned selected options result
        2. "presence" : returned selected options' top 10 presence words in real and absence words in fake
        3. "absence" : returned selected options' top 10 presence words in fake and absence words in real
        4. "specific" : returned specific words for real and fake datas"""
        
#All_Results(2)
""" some examples in here """
#main(1,stopWords="english",Stemming=False,tfidf=False,PrintCommand="General Results")
#main(1,stopWords="english",Stemming=False,tfidf=False,PrintCommand="specific")
#main(1,stopWords="english",Stemming=False,tfidf=True,PrintCommand="presence")
#main(1,stopWords="english",Stemming=False,tfidf=True,PrintCommand="absence")
