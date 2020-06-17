# coding: utf-8
import sys
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from datetime import datetime

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
# ファイル名の指定 (頻出語抽出データを対象とする)
#------------------------------------------------------------
# 対象ファイル名
current_dir = "."
file_name = "all_text_data"

start_datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

print("\nCreate Model -> START")

# パスの指定
base_dir = "/mnt/ext01/lecturer_similar"

# 分かち書きファイル名
wakati_name = base_dir + "/text_file/mc_wakati_" + file_name + ".txt"
print(wakati_name)

# モデルファイル名
model_name = base_dir + "/model/" + "mc_" + file_name + model_suffix + ".model"

#------------------------------------------------------------
# テキストデータから学習データを設定
#------------------------------------------------------------
# テキストデータを取得
text = open(wakati_name, 'r',  encoding="utf-8").read()

# 講師情報は ### で区切られている
documents = text.split("###")

print("lecturer num : " , len(documents) - 1, "\n")

# 学習データの設定
training_docs = []
for i, document in enumerate(documents):
    # lecturer_id とテキストは &&& で区切られている
    lecturer_info = document.split("&&&")
    if len(lecturer_info) >= 2:
        lecturer_id = lecturer_info[0].strip()
        print("lecturer_id : ", lecturer_id)
        training_docs.append(TaggedDocument(words=lecturer_info[1], tags=[str(lecturer_id)]))
        
    else:
        print("")
        print(lecturer_info)

print("\lecturer num : " , len(documents) - 1, "\n")

#sys.exit()

print(model_name, "\n")

#------------------------------------------------------------
# モデルの作成とファイル保存
#------------------------------------------------------------
print("START :" , start_datetime, "\n");

# モデル作成 (dm=1: PV-DM dmpv)
model = Doc2Vec(documents=training_docs, dm=1, min_count=param_min_count, epochs=param_epochs)
#model = Doc2Vec(documents=training_docs, min_count=1, dm=0, epochs=50)
#model = Doc2Vec(documents=training_docs, dm=1)

# モデルのセーブ
model.save(model_name)

print(model_name, " を作成しました。")

end_datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S");
print("END :" , end_datetime, "\n");
print("Create Model -> END (", start_datetime, "-", end_datetime, ")\n")

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
