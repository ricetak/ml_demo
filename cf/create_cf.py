# coding: utf-8
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# パスの指定
base_dir = "."

print("\n\n---------- START ----------")

#-----------------------------------------------------
# パラメータの初期化
#-----------------------------------------------------
user_min = 20
k= 35

#-----------------------------------------------------
# 引数が渡された場合の処理
#-----------------------------------------------------
args = sys.argv
if len(args) == 2:
    user_min = args[1]
elif len(args) == 3:
    user_min = args[1]
    k = args[2]
#else:
    #print('\n以下のような形式で[user_min]と[k]を指定してください')
    #print('python3 \nmf_test.py 20 35 \n')

user_min = int(user_min)
k = int(k)

print("user_min = " + str(user_min))
print("k = " + str(k))
print()

#-----------------------------------------------------
# CSVファイルをpandas.DataFrame に読み込む
#-----------------------------------------------------
df = pd.read_csv(base_dir + '/data/score.csv')
#print(df)
#sys.exit()


#-----------------------------------------------------
# 対象データの整形
#-----------------------------------------------------
# id 列をインデックスに指定
df_i = df.set_index('id')
#print(df_i.index)

# [id]指定した本数以上のユーザーのみ(1以上の要素を true として合算する)
df_bool = (df_i > 0)
#print(df_bool)
#print(df_bool.sum(axis=1))
indexer_user = df_bool.sum(axis=1) >= user_min
df_i_user = df_i[indexer_user]
print(df_i_user)

# [講義ID]全てが0の列を除去
indexer_movie_id = df_i_user.sum(axis=0) > 0
df_target = df_i_user[indexer_movie_id.index[indexer_movie_id]]
print(df_target)

#-----------------------------------------------------
# データ妥当性のチェック
#-----------------------------------------------------
# 0のみの行の検知
print("#####################################")
indexer_zero = df_target.sum(axis=1) == 0
print(df_target[indexer_zero])
print(len(df_target[indexer_zero]))
print()

# 0のみの列の検知
indexer_zero = df_target.sum(axis=0) == 0
print(df_target[indexer_zero.index[indexer_zero]])
print(len(df_target[indexer_zero.index[indexer_zero]].columns))

#-----------------------------------------------------
# カラムとインデックスの保持
#-----------------------------------------------------
score_columns = df_target.columns
score_index = df_target.index

#print(score_columns)
#print(score_index)

#-----------------------------------------------------
# nmf 開始
#-----------------------------------------------------
print("\n*************************")
print("START nmf")
print("*************************")

Y = df_target.values
n, m = Y.shape
W = np.random.rand(n, k)
H = np.random.rand(k, m)
mask_Y = (Y > 0)

print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#**************************************
# multiplicative update rules
#**************************************
def update(Y, W, H, mask):
    W = W * np.dot(Y * mask, H.T) / np.dot(np.dot(W, H) * mask, H.T)
    H = H * np.dot(W.T, Y * mask) / np.dot(W.T, np.dot(W, H) * mask)
    return W, H

# WとHの更新を繰り返してWとHの行列の積を徐々にYに近づけていく
for _ in range(100):
    W, H = update(Y, W, H, mask_Y)
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#-----------------------------------------------------
# CSVファイルへ出力
#-----------------------------------------------------
nm_csv_file = base_dir + "/data/nmf_score_" + str(user_min) + "_"  + str(k)  +".csv"

#df = pd.DataFrame(W @ H, index=score_index, columns=score_columns)
df = pd.DataFrame(np.dot(W, H), index=score_index, columns=score_columns)
print(df)

df.to_csv(nm_csv_file)

print()
print(nm_csv_file + " を作成しました")

print("\n\n---------- END ----------\n\n")

sys.exit()

