# coding: utf-8
import sys
import os
import MeCab

#------------------------------------------------------------
# テキストデータを MeCabで分かち書きする
#------------------------------------------------------------
# 対象データファイル
file_name = "all_text_data"

print("\nMeCab -> START")

# 分かち書きファイル名を生成
wakati_name = "./text_file/wakati_" + file_name + ".txt"

# 分かち書き
tagger = MeCab.Tagger("-Owakati")

#tagger.parse("")
all_words = "";
for line in open("./text_file/" + file_name + ".txt", "r", encoding="utf-8"):
    words = tagger.parse(line)
    all_words = all_words + words

#print(all_words)

f = open(wakati_name,'w', encoding='utf-8')
f.write(all_words)
f.close()

print()
print(wakati_name, " を作成しました。")
print()
print("MeCab -> END\n")

sys.exit()
