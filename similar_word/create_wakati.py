# coding: utf-8
import sys
import os
import MeCab

#------------------------------------------------------------
# wakati text data with MeCab
#------------------------------------------------------------
# data file
file_name = "kokoro"

print("\nMeCab -> START")

# Create file name
wakati_name = "./text_file/wakati_" + file_name + ".txt"

# wakati
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
print("save to " +  wakati_name)

print()
print("MeCab -> END\n")

sys.exit()
