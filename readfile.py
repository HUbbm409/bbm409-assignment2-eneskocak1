import pandas as pd


##########################################################################################################
def readfile(real,fake,test):
    ##############################################
    #    READING TEST,REAL and FAKE DATAS        #
    #         AND SPLITTING LINES                #
    ##############################################
    realfile = open(real,"r")
    realstring = realfile.read().splitlines() 
    realfile.close()
    fakefile = open(fake,"r")
    fakestring = fakefile.read().splitlines() 
    fakefile.close()
    #print(len(realstring),len(fakestring))
    testfile = pd.read_csv(test,sep=",",encoding="latin-1",error_bad_lines=False,warn_bad_lines=False,low_memory=False)
    testfile.columns=["line","class"]
    trueclassifedlist=testfile["class"].tolist()
    teststring = testfile["line"].tolist() 
    #print(len(realstring),len(fakestring),len(teststring))
    return realstring,fakestring,teststring,trueclassifedlist
###########################################################################################################
    