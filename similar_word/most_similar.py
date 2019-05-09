# coding: utf-8
import sys
from gensim.models import word2vec

def most_similar(target_word, file_name):
    # 対象情報を表示
    print("\n対象ワード : " + target_word)
    print("対象モデル : " + file_name)
    print("\n### most_similar START : " + file_name + " <- "  + target_word + "\n")

    # モデルを指定
    model_name = "./model/" + file_name + ".model"
    
    # Word2Vec のモデルを読み込む
    model = word2vec.Word2Vec.load(model_name)
    print(model)
    #print(model.wv)
    print()
    
    # KeyedVectors を取得
    word_vectors = model.wv
    
    # モデルを削除 (メモリ節約)
    del model

    # 類義語を取得
    try:
        #results = model.wv.most_similar(positive=[target_word])
        results = word_vectors.most_similar(positive=[target_word], topn=20)
        for result in results:
            print(result)
        
        #print()
        #results = word_vectors.most_similar_cosmul(positive=[target_word], topn=10)
        #for result in results:
        #    print(result)
            
    except:
        print('このモデルには', target_word, 'の類義語は無いようです')
    
    # 終了
    print("\n### most_similar END\n")


# 対象モデルファイル
file_name = "kokoro"

# 対象ワード
target_word = '下宿'

# 引数が渡された場合の処理
args = sys.argv
if len(args) == 2:
    target_word = args[1]
    
elif len(args) == 3:
    target_word = args[1]
    file_name = args[2]

else:
    print('\n以下のような形式で[ワード]と[モデル]を指定してください')
    print('python3 most_similar.py 再生 all_text_data \n')

# 類義語取得
most_similar(target_word, "kokoro")

