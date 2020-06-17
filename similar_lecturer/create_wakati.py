# coding: utf-8
import sys
import os
import MeCab
from datetime import datetime

#------------------------------------------------------------
# テキストデータを MeCabで分かち書きする
#------------------------------------------------------------
# 対象データファイル
file_name = "all_text_data"

start_datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

print("\nMeCab -> START")

# パスの指定
base_dir = "/mnt/ext01/lecturer_similar"

# 対象テキストファイル名を生成
text_file_name = base_dir + "/text_file/" + file_name + ".txt"

# 分かち書きファイル名を生成
wakati_name = base_dir + "/text_file/wakati_" + file_name + ".txt"

print(text_file_name)
print(wakati_name)

# 分かち書き
tagger = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

all_words = "";
for line in open(text_file_name, "r", encoding="utf-8"):
    words = tagger.parse(line)
    all_words = all_words + words

#print(all_words)

f = open(wakati_name,'w', encoding='utf-8')
f.write(all_words)
f.close()

print()
print(wakati_name, " を作成しました。")
print()
#print("MeCab -> END\n")

end_datetime = datetime.now().strftime("%Y/%m/%d %H:%M:%S");
print("MeCab -> END (", start_datetime, "-", end_datetime, ")\n")

sys.exit()
