# coding: utf-8
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt   # 追加
import numpy as np
import random
import sys
import io
from datetime import datetime as dt

class TG:
    
    #*******************************************************************
    # コンストラクタ
    #*******************************************************************
    # target_text : 対象テキストファイル名
    # maxlen : いくつの「字」を1つの「文」とするか   
    # create_len : 生成する文字数
    # start_index_flag : 0の場合は文頭から文章を生成, 0以外の場合はランダム
    # epochs : 学習回数
    #*******************************************************************
    def __init__(self, target_text, maxlen = 8, create_len = 1000, start_index_flag = 0, epochs = 60):
       
        self.target_text = target_text
        self.maxlen = maxlen
        self.create_len = create_len
        self.start_index_flag = start_index_flag
        self.epochs = epochs

        self.file_process()
        self.main_process()
      

    #*******************************************************************
    # ファイル関連処理
    #*******************************************************************
    def file_process(self):
        
        target_text = self.target_text
        
        current_datetime = dt.now()
        #current_datetime_str = current_datetime.strftime('%Y/%m/%d %H:%M:%S')
        current_datetime_str = current_datetime.strftime('%Y%m%d_%H%M%S')
        
        self.path = './inter_data/' + target_text + '_data.txt'
        self.outfile_path = './output/' + target_text + '_out_' + current_datetime_str + '.txt'
        self.outfile = open(self.outfile_path,'w',encoding='utf-8')
        
        print("data file : ", self.path)
        print("out file : ", self.outfile_path, "\n")
        #sys.exit()
        
        with io.open(self.path, encoding='utf-8') as f:
            self.text = f.read().lower()
            
        print('corpus length:', len(self.text))
        print('corpus length:', len(self.text), file=self.outfile)

    #*******************************************************************
    # メイン処理
    #*******************************************************************
    def main_process(self):
        
        maxlen = self.maxlen
        
        text = self.text
        outfile = self.outfile

        #---------------------------------------------------------
        # 各々の字のdict作成
        #---------------------------------------------------------
        # chars : 重複を排除した「字」のリスト
        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        print('total chars:', len(chars), file=outfile)
        
        # char_indices : 「字」を上記charsのindex番号に変換するdict
        char_indices = dict((c, i) for i, c in enumerate(chars))
        
        # indices_char : 上記と逆にindex番号を「字」に変換するdict
        indices_char = dict((i, c) for i, c in enumerate(chars))

        #---------------------------------------------------------
        # 学習用データに整形
        #---------------------------------------------------------
        step = 1
        
        # sentences  : 「文」のリスト
        sentences = []
        
        # next_chars : 各「文」について、その次の「文」の最初の「字」
        next_chars = []
        
        for i in range(0, len(text) - maxlen, step):
            # 長さで区切った部分文字列を一つの文という扱いで抽出
            sentences.append(text[i: i + maxlen])
            
            # 次の文の最初の文字を保存
            next_chars.append(text[i + maxlen])
            
        # 上記の「文」の数をそのままLSTMのsequence数として用いる
        print('nb sequences:', len(sentences))
        print('nb sequences:', len(sentences), file=outfile)
        print("\n", file=outfile)
        
        print()
        print('Vectorization...')
        
        # x : np.bool型 3次元配列 [文の数, 文の最大長, 字の種類] 
        #     -> 文中の各位置に各indexの文字が出現するか
        x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
        
        # y : np.bool型 2次元配列 [文の数, 字の種類] 
        #     -> 次の文の開始文字のindex
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        
        # vector化は各「文」について実施
        for i, sentence in enumerate(sentences):
            
            print("-----------------------------------")
            print(i, " : " , sentence)
            
            for t, char in enumerate(sentence):
                        #print(t, " : " , char_indices[char])
                
                x[i, t, char_indices[char]] = 1
                        
                print(t, ":" , char, "x[" ,i, t, char_indices[char], "] = 1");
        
                
            y[i, char_indices[next_chars[i]]] = 1
            #print(i, " : " , next_chars[i])
            #print("y[" , i, char_indices[next_chars[i]], "] = 1");
            
            print(i, " : " , next_chars[i], "y[" , i, char_indices[next_chars[i]], "] = 1");

        #sys.exit()
        
        #---------------------------------------------------------
        # モデル作成
        #---------------------------------------------------------
        # build the model: a single LSTM
        print('Build model...')
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))
        
        # 勾配法にRMSpropを用いる
        optimizer = RMSprop(lr=0.01)
        
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        self.chars = chars
        self.char_indices = char_indices
        self.indices_char = indices_char
        
        self.model = model
        
        #---------------------------------------------------------
        # 各epoch終了時のcallbackとして、上記のon_epoch_endを呼ぶ
        #---------------------------------------------------------
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        
        #---------------------------------------------------------
        # フィッティング :各epoch完了時に on_epoch_end が呼ばれる
        #---------------------------------------------------------]
        history = model.fit(
                        x, y,
                        batch_size=128,
                        #epochs=60,
                        #epochs=80,
                        epochs=self.epochs,
                        callbacks=[print_callback])
        
        #---------------------------------------------------------
        # Plot Training loss & Validation Loss
        #---------------------------------------------------------
        loss = history.history["loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label = "Training loss" )
        plt.title("Training loss")
        plt.legend()
        plt.savefig("loss.png")
        plt.close()
        
        
        outfile.close()
        
        print()
        print("data file : ", self.path)
        print("out file : ", self.outfile_path)
        
        print("\n\n### END ### \n\n")

    #---------------------------------------------------------
    # 出力する文字をランダムに選ぶ
    #---------------------------------------------------------
    def get_next_char_index(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        
        return np.argmax(probas)


    #*******************************************************************
    # 各エポックの終了時にその時点のモデルを使ったテキスト生成処理
    #*******************************************************************
    def on_epoch_end(self, epoch, logs):
        
        outfile = self.outfile
        text = self.text
        maxlen = self.maxlen
        
        print('*************************************************************', file=outfile)
        print('* Generating text after Epoch: %d' % epoch, file=outfile)
        print('*************************************************************', file=outfile)
            
        if self.start_index_flag == 0:
            start_index = 0
        else:
            start_index = random.randint(0, len(text) - maxlen - 1)
       
        print("start_index : ", start_index)
       
        #for diversity in [0.2, 0.5, 1.0, 1.2]:
        for diversity in [0.2, 1.0]:
            print('----- diversity:', diversity)
            print('----- diversity:', diversity, file=outfile)
    
            generated = ''
            
            # 元にする「文」を選択
            sentence = text[start_index: start_index + maxlen]
            
            generated += sentence
            #print('----- Generating with seed: "' + sentence + '"')
            #sys.stdout.write(generated)
            
            generated_for_file = "";
            
            # 生成する文字数
            #create_len = 1000
            create_len = self.create_len
            
            chars = self.chars
            char_indices = self.char_indices
            model = self.model
            
            # 「文」に続くcreate_len個の「字」をモデルから予測し出力する
            for i in range(create_len):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                     # 最初の次元は「文」のIDなので0固定
                    x_pred[0, t, char_indices[char]] = 1. # 数値がピリオド"."で終了する場合浮動小数点
    
                # 現在の「文」に続く「字」を予測する
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = self.get_next_char_index(preds, diversity)
                next_char = self.indices_char[next_index]
    
                # 予測して得られた「字」を生成し、「文」に追加
                generated += next_char
                
                # 例：sentence 「これはメイドインジャパン」
                #     next_char 「の」-> sentence 「れはメイドインジャパンの」
                sentence = sentence[1:] + next_char

            generated_for_file += "\n" + generated + "\n\n"
            outfile.write(generated_for_file)
             
            #print()
            

model = TG("roll", maxlen = 8, create_len = 1000, start_index_flag = 0, epochs = 60)

