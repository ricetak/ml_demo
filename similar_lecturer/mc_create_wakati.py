# coding: utf-8
import sys
import os
import MeCab
from datetime import datetime

import collections  # 単語数カウント用

#------------------------------------------------------------
# テキストデータを MeCabで分かち書きする : 頻出語抽出版
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
wakati_name = base_dir + "/text_file/mc_wakati_" + file_name + ".txt"

print(text_file_name)
print(wakati_name)

# 分かち書き
tagger = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

# テキストデータを取得
text = open(text_file_name, 'r',  encoding="utf-8").read()

# 講師情報は ### で区切られている
documents = text.split("###")

print("lecturer num : " , len(documents) - 1, "\n")

all_words = "";

for i, document in enumerate(documents):
     # lecturer_id とテキストは &&& で区切られている
    lecturer_info = document.split("&&&")
    if len(lecturer_info) >= 2:
        lecturer_id = lecturer_info[0].strip()
        print("lecturer_id : ", lecturer_id)
        
        all_words = all_words + lecturer_id + " &&& "
        
        if int(lecturer_id) <= 9999: 
            node = tagger.parseToNode(lecturer_info[1])
            words=[]
            while node:
                hinshi = node.feature.split(",")[0]
                #print(hinshi)
                if hinshi in ["名詞", "動詞", "形容詞"]:
                    origin = node.feature.split(",")[6]
                    words.append(origin)
                node = node.next
                
            c = collections.Counter(words)
            #print(c.most_common())
            
            # 頻度順に出力
            counter = collections.Counter(words)
            #print(counter)
            for word, count in counter.most_common(300):
                #print(f"{word}: {count}")
                all_words = all_words + " " + word
                
            all_words = all_words + " ### \n";

#print(all_words)
  
print("\lecturer num : " , len(documents) - 1, "\n")
 
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
