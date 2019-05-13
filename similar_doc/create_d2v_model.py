# coding: utf-8
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

#------------------------------------------------------------
# パラメータ
#------------------------------------------------------------
param_min_count = 1
param_epochs = 50

#------------------------------------------------------------
# 引数が渡された場合の処理
#------------------------------------------------------------
args = sys.argv
if len(args) == 2:
    param_min_count = args[1]
    
elif len(args) == 3:
    param_min_count = args[1]
    param_epochs = args[2]

else:
    print('\n以下のような形式で[min_count]と[epochs]を指定してください')
    print('python3 create_d2v_model_only.py 1 50 \n')

print("param_min_count = ",  param_min_count)
print("param_epochs = ",  param_epochs)

param_min_count = int(param_min_count)
param_epochs = int(param_epochs)

model_suffix = ""
if len(args) == 3:
    model_suffix = "_" + str(param_min_count) + "_" + str(param_epochs)

#------------------------------------------------------------
# ファイル名の指定
#------------------------------------------------------------
# 対象ファイル名
current_dir = "."
file_name = "all_text_data"

print("\nCreate Model -> START")

# 分かち書きファイル名
wakati_name = "./text_file/wakati_" + file_name + ".txt"
print(wakati_name)

# モデルファイル名
model_name = current_dir + "/model/" + file_name + model_suffix + ".model"

#------------------------------------------------------------
# テキストデータから学習データを設定
#------------------------------------------------------------
# テキストデータを取得
text = open(wakati_name, 'r',  encoding="utf-8").read()

# 講義情報は ### で区切られている
documents = text.split("###")

print("article num : " , len(documents), "\n")

# 学習データの設定
training_docs = []
for i, document in enumerate(documents):
    # movie_idとテキストは &&& で区切られている
    movie_info = document.split("&&&")
    if len(movie_info) >= 2:
        move_id = movie_info[0].strip()
        print("move_id : ", move_id)
        training_docs.append(TaggedDocument(words=movie_info[1], tags=[str(move_id)]))
         
    else:
        print("")
        print(movie_info)

print("\narticle num : " , len(documents), "\n")

#------------------------------------------------------------
# create model and to save file
#------------------------------------------------------------
# create model (dm=0: PV-DBOW)
#model = Doc2Vec(documents=training_docs, min_count=1, dm=0, epochs=50)

# create model (dm=1: PV-DM dmpv)
model = Doc2Vec(documents=training_docs, dm=1, min_count=param_min_count, epochs=param_epochs)
#model = Doc2Vec(documents=training_docs, dm=1)

# save file
model.save(model_name)

print(model_name, " を作成しました。")

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


