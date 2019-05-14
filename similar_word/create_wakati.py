# coding: utf-8
import MeCab

# data file
file_name = "kokoro"

print("### create wakati file : " + file_name)

#################################################################
# wakati text data with MeCab
#################################################################
print("MeCab -> START")

# Create file name
wakati_name = "./text_file/wakati_" + file_name + ".txt"

# Get text data
target_file = open("./text_file/" + file_name + ".txt", "r", encoding="utf-8")
target_text = target_file.read()
target_file.close()

# wakati
tagger = MeCab.Tagger("-Owakati")

f = open(wakati_name,'w', encoding='utf-8')

tagger.parse("")

f.write(tagger.parse(target_text))
f.close()

print()
print(wakati_name)
print()
print("MeCab -> END")
