
from nltk.stem.porter import PorterStemmer
import nltk
import pandas as pd


def list_tokenizer(reallist,fakelist,testlist,BoW,steamming):
    
   
    porter_stemmer = PorterStemmer()   
    for i in range(len(reallist)):
        # words was stemming here when stemming == true;
        if steamming==True:
            nltk_tokens = nltk.word_tokenize(reallist[i])
            new_string=""
            for w in nltk_tokens:
                new_string += porter_stemmer.stem(w)+" "
            reallist[i]=new_string
         # when n_gram == 2 ; adding token start of line and end of line 
        if BoW==2:
            reallist[i]="_s_ " + reallist[i]+ " _eos_"
        
    for i in range(len(fakelist)):
        if steamming==True:
            nltk_tokens = nltk.word_tokenize(fakelist[i])
            new_string=""
            for w in nltk_tokens:
                new_string += porter_stemmer.stem(w)+" "
            fakelist[i]=new_string
        if BoW==2:
            fakelist[i]="_s_ " + fakelist[i]+ " _eos_"
        
    for i in range(len(testlist)):
        if steamming==True:
            nltk_tokens = nltk.word_tokenize(testlist[i])
            new_string=""
            for w in nltk_tokens:
                new_string += porter_stemmer.stem(w)+" "
            testlist[i]=new_string
        if BoW==2:
            testlist[i]="_s_ " + testlist[i]+ " _eos_"
    
    #print(reallist[0])
  


#################################################################################################
def dedect_specific_words(real_word_matrix,real_word_id_dict,fake_word_matrix,fake_word_id_dict,specific=None):
    ##############################################
    #                  PART 1                    #
    # Detecting 3 specific word for clustering   #
    # AND PART3 RESULTS DETECTED IN WORD_HISTOGRAM
    ##############################################
  
    real_word_frequencies = real_word_matrix.sum(axis=0)
    real_words_freq = [(word, real_word_frequencies[0, idx]) for word, idx in real_word_id_dict.items()]
    
    fake_word_frequencies = fake_word_matrix.sum(axis=0)
    fake_words_freq = [(word, fake_word_frequencies[0, idx]) for word, idx in fake_word_id_dict.items()]

    
    
    real_word_dict=dict(real_words_freq)
    fake_word_dict=dict(fake_words_freq)
    if specific=="specific":
        word_histogram(real_word_dict,fake_word_dict,3,Type=specific)
    if specific=="presence":
        word_histogram(real_word_dict,fake_word_dict,10,Type=specific)
    if specific=="absence":
        word_histogram(real_word_dict,fake_word_dict,10,Type=specific)
    
    return real_word_dict,fake_word_dict
   
    
####################################################################################################   
    
####################################################################################################   


def word_histogram(realwords,fakewords,wordnumber,Type=None):
    realdict={}
    fakedict={}
    
    # detecting deiffrence sets real and fake class's #
    for (item,value) in realwords.items():
        if item not in fakewords.keys():
            realdict[item]=value
    reallist= list(realdict.items())
    reallist =sorted(reallist,key=lambda l:l[1], reverse=True)
    
    for item,value in fakewords.items():
        if item not in realwords.keys():
            fakedict[item]=value
    # sorting by frequency
    # this is same with conditional probailtys. because the denaminators are the same 
    fakelist= list(fakedict.items())
    fakelist =sorted(fakelist,key=lambda l:l[1], reverse=True)
    
    realxbar=[]
    realybar=[]
    fakexbar=[]
    fakeybar=[]
    
    if Type=="presence":

        print("\n10 words whose presence most strongly predicts that the news is real.\nAnd whose absence most strongly predicts that the news is fake.\n")
        for x in range(wordnumber):
            realxbar.append(reallist[x][0])
            realybar.append(reallist[x][1])
        r = pd.DataFrame({"Word/WordPairs":realxbar, 'Frequency':realybar})
        print(r)

    if Type=="specific":    
        
        print(wordnumber,"Specific keys for detect real news:")
        for x in range(wordnumber):
            realxbar.append(reallist[x][0])
            realybar.append(reallist[x][1])
        r = pd.DataFrame({"Word/WordPairs":realxbar, 'Frequency':realybar})
        print(r)
        
        print("\n\n",wordnumber,"Specific keys for detect fake news:")
        for x in range(wordnumber):
            fakexbar.append(fakelist[x][0])
            fakeybar.append(fakelist[x][1])
        f = pd.DataFrame({"Word/WordPairs":fakexbar, 'Frequency':fakeybar})
        print(f)
    
   
    if Type=="absence":
 
        print("\n10 words whose absence most strongly predicts that the news is real.\nAnd whose presence most strongly predicts that the news is fake.\n")
        for x in range(wordnumber):
            fakexbar.append(fakelist[x][0])
            fakeybar.append(fakelist[x][1])
        f = pd.DataFrame({"Word/WordPairs":fakexbar, 'Frequency':fakeybar})
        print(f)

  

    
        
    

