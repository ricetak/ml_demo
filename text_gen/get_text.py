
import sys
import re
import io

target_text = "roll"
text_path = './data/'  + target_text + '.txt'
data_path = './inter_data/' + target_text + '_data.txt'

print()
print("text_path : ", text_path)
print("data_path : ", data_path)

data_file = open(data_path,'w',encoding='utf-8')

bindata = open(text_path, "rb")
lines = bindata.readlines()

print("\n########## START ##########")
for line in lines:
    text = line.decode('utf-8')        
    text = re.split(r'\r',text)[0]         
    text = text.replace('　', ' ')
    text = text.replace('●', '')
    
    print(text)
    data_file.write(text)

print("\n########## END ##########")

data_file.close()


text_file = open(text_path,'r',encoding='utf-8')
all_text = text_file.read()
print()
print('all_text length:', len(all_text))

data_file = open(data_path,'r',encoding='utf-8')
all_text_data = data_file.read()
print('all_inter_text_data length:', len(all_text_data))

print()
print("text_path : ", text_path)
print("data_path : ", data_path)
