# coding: utf-8
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

#------------------------------------------------------------
# parameter
#------------------------------------------------------------
param_min_count = 1
param_epochs = 50

#------------------------------------------------------------
# Set file name
#------------------------------------------------------------
# file anme
current_dir = "."
file_name = "all_text_data"

print("\nCreate Model -> START")

# wakati file name
wakati_name = "./text_file/wakati_" + file_name + ".txt"
print(wakati_name)

# model file name
model_name = current_dir + "/model/" + file_name + ".model"

#------------------------------------------------------------
# Set learning data from text data
#------------------------------------------------------------
# Get text data
text = open(wakati_name, 'r',  encoding="utf-8").read()

# Article information is separated by ###
documents = text.split("###")

print("article num : " , len(documents), "\n")

# Set learning data
training_docs = []
for i, document in enumerate(documents):
    # Article id and text content are separated by &&&
    article_info = document.split("&&&")
    if len(article_info) >= 2:
        article_id = article_info[0].strip()
        print("article_id : ", article_id)
        training_docs.append(TaggedDocument(words=article_info[1], tags=[str(article_id)]))
         
    else:
        print("")
        print(article_info)

print("\narticle num : " , len(documents), "\n")

#------------------------------------------------------------
# create model and to save file
#------------------------------------------------------------
# create model (dm=0: PV-DBOW)
#model = Doc2Vec(documents=training_docs, min_count=1, dm=0, epochs=50)

# create model (dm=1: PV-DM dmpv)
model = Doc2Vec(documents=training_docs, dm=1, min_count=param_min_count, epochs=param_epochs)

# save file
model.save(model_name)

print("save to " , model_name)

print("Create Model -> END\n")

sys.exit()

'''
Doc2Vec(
    documents=None, 
    corpus_file=None, 
    dm_mean=None, 
    dm=1, 
    dbow_words=0, 
    dm_concat=0, 
    dm_tag_count=1, 
    docvecs=None, 
    docvecs_mapfile=None, 
    comment=None, 
    trim_rule=None, 
    callbacks=(), 
    **kwargs)

BaseWordEmbeddingsModel(
    sentences=None, 
    corpus_file=None, 
    workers=3, 
    vector_size=100, 
    epochs=5, 
    callbacks=(), 
    batch_words=10000, 
    trim_rule=None, 
    sg=0, 
    alpha=0.025, 
    window=5, 
    seed=1, 
    hs=0, 
    negative=5, 
    ns_exponent=0.75, 
    cbow_mean=1, 
    min_alpha=0.0001, 
    compute_loss=False, 
    fast_version=0, 
    **kwargs)
''' 


