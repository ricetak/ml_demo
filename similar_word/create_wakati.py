# coding: utf-8
import MeCab

# 対象データファイル
file_name = "kokoro"
#file_name = "lec_1"

print("### create wakati file : " + file_name)

#################################################################
# テキストデータを MeCabで分かち書きする
#################################################################
print("MeCab -> START")

# 分かち書きファイル名を生成
wakati_name = "./text_file/wakati_" + file_name + ".txt"

# テキストデータを取得
target_file = open("./text_file/" + file_name + ".txt", "r", encoding="utf-8")
target_text = target_file.read()
target_file.close()

# 分かち書き
tagger = MeCab.Tagger("-Owakati")

f = open(wakati_name,'w', encoding='utf-8')

tagger.parse("")

f.write(tagger.parse(target_text))
f.close()

print()
print(wakati_name)
print()
print("MeCab -> END")
