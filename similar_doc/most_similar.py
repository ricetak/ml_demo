# coding: utf-8
import sys
import os
from gensim.models.doc2vec import Doc2Vec

#------------------------------------------------------------
# get most_similar list
#------------------------------------------------------------
def most_similar(target_id, file_name):
    
    model_name = "./model/" + file_name + ".model"
 
    # Load Doc2Vec Model
    model = Doc2Vec.load(model_name)
    
    try:  
        items = model.docvecs.most_similar(target_id, topn=50)
        for item in items:
            print(item)
         
    except:
        print("-1") 
        

model_name = "all_text_data"

target_id = '3'

most_similar(target_id, model_name)

sys.exit()

