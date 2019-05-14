# coding: utf-8
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# path
base_dir = "."

print("\n\n---------- START ----------")

#-----------------------------------------------------
# parameter
#-----------------------------------------------------
user_min = 20
k= 35

#-----------------------------------------------------
# Load CSV file into pandas.DataFram
#-----------------------------------------------------
df = pd.read_csv(base_dir + '/data/score.csv')

#-----------------------------------------------------
# Reshape the data
#-----------------------------------------------------
# Setid column as index
df_i = df.set_index('id')

# Sum one or more elements as true
df_bool = (df_i > 0)

# Users who watched more than the specified number
indexer_user = df_bool.sum(axis=1) >= user_min
df_i_user = df_i[indexer_user]
print(df_i_user)

# Remove columns with all values 0
indexer_movie_id = df_i_user.sum(axis=0) > 0
df_target = df_i_user[indexer_movie_id.index[indexer_movie_id]]
print(df_target)

#-----------------------------------------------------
# Check data validity
#-----------------------------------------------------
# Detect rows with all values 0
print("#####################################")
indexer_zero = df_target.sum(axis=1) == 0
print(df_target[indexer_zero])
print(len(df_target[indexer_zero]))
print()

# Detect columns with all values 0
indexer_zero = df_target.sum(axis=0) == 0
print(df_target[indexer_zero.index[indexer_zero]])
print(len(df_target[indexer_zero.index[indexer_zero]].columns))

#-----------------------------------------------------
# Keep columns and indexes
#-----------------------------------------------------
score_columns = df_target.columns
score_index = df_target.index

#-----------------------------------------------------
# nmf
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

# Repeat W and H updates and Make the product of W and H matrices gradually closer Y
for _ in range(100):
    W, H = update(Y, W, H, mask_Y)
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#-----------------------------------------------------
# save to CSV file
#-----------------------------------------------------
nm_csv_file = base_dir + "/data/nmf_score_" + str(user_min) + "_"  + str(k)  +".csv"

#df = pd.DataFrame(W @ H, index=score_index, columns=score_columns)
df = pd.DataFrame(np.dot(W, H), index=score_index, columns=score_columns)
print(df)

df.to_csv(nm_csv_file)

print()
print("save to " + nm_csv_file)

print("\n\n---------- END ----------\n\n")

sys.exit()

