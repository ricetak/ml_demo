# coding: utf-8
import sys
import os
from gensim.models.doc2vec import Doc2Vec

#------------------------------------------------------------
# モデルを読み込み類似講義リストを取得する
#------------------------------------------------------------
def most_similar(target_id, file_name):
    # 対象情報を表示
    #sys.stderr.write("\n対象講義ID : " + target_id)
    #sys.stderr.write("対象モデル : " + file_name)
    #sys.stderr.write("\n### most_similar START : " + file_name + " <- "  + target_id + "\n")

    # モデルを指定
    if os.name == "nt" :
        # ローカルPC(Windows)で実行した場合
        model_name = "./model/" + file_name + ".model"
    else :
        # ローカルPC以外(Windows以外)で実行した場合
        base_dir = "/mnt/ext01/tmp/doc_similar";
        model_name = base_dir + "/model/" + file_name + ".model"
 
    # Doc2Vec のモデルを読み込む
    model = Doc2Vec.load(model_name)
    
    # 類似講義を取得
    try:  
        items = model.docvecs.most_similar(target_id, topn=50)
        for item in items:
            print(item)
         
    except:
        print("-1")
        #print('このモデルには', target_id, 'の類義講義は無いようです')
        
    #sys.stderr.write("\n### most_similar END\n")

#------------------------------------------------------------
# 対象モデルファイル
#------------------------------------------------------------
model_name = "all_text_data"

# 対象講義ID
target_id = '3'

#------------------------------------------------------------
# 引数が渡された場合の処理
#------------------------------------------------------------
args = sys.argv
if len(args) == 2:
    target_id = args[1]
    
elif len(args) == 3:
    target_id = args[1]
    model_name = args[2]

else:
    print('\n以下のような形式で[講義ID]と[モデル]を指定してください')
    print('python3 most_similar.py 1639 all_text_data \n')

#------------------------------------------------------------
# 類似講義取得
#------------------------------------------------------------
most_similar(target_id, model_name)

sys.exit()

