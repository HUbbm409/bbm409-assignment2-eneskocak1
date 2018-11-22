
import pandas as pd

###########################################################################################################
def Accuracy(truelist,classifiedlist,PrintCommand=None):
    
    
    correct_classifed_item=0
    for i in range(len(truelist)):
        if truelist[i]==classifiedlist[i]:
            correct_classifed_item+=1
   
    accuracy=(correct_classifed_item/len(truelist))*100
    if PrintCommand=="General Results":
        print("Correct classified news:",correct_classifed_item)
    return accuracy,correct_classifed_item


