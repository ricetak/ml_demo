# coding: utf-8
from gensim.models import word2vec

# 対象データファイル
file_name = "kokoro"

# 各ファイル名を生成
wakati_name = "./text_file/wakati_" + file_name + ".txt"
model_name = "./model/" + file_name + ".model"

print("### create Model START : " + file_name)

print(wakati_name)

#################################################################
# Word2Vec のモデル作成
#################################################################
print("Word2Vec create Model -> START")

#sentences = word2vec.Text8Corpus(wakati_name)
sentences = word2vec.LineSentence(wakati_name)

model = word2vec.Word2Vec(sentences, sg=1, size=100, window=10, iter=10, min_count=5)

model.save(model_name)

print("Word2Vec create Model -> END")

#################################################################

print("")
print(model_name + " を作成しました")
print("")

print("### create Model END")


'''
word2vec.Word2Vec(
    sentences=None, size=100, alpha=0.025, window=5, min_count=5, 
    max_vocab_size=None, sample=0.001, seed=1, workers=3, 
    min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75, 
    cbow_mean=1, hashfxn=<built-in function hash>, 
    iter=5, null_word=0, trim_rule=None, sorted_vocab=1,
    batch_words=10000, compute_loss=False, callbacks=(), 
    max_final_vocab=None)
'''


