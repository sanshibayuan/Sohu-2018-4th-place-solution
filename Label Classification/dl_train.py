import numpy as np, pandas as pd
from sklearn.externals import joblib
import pickle
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D,BatchNormalization
from dl_models import *
from keras.utils import to_categorical, normalize

train =pd.read_table('News_info_train_seg.txt',sep =  '\t' , header = None)
test = pd.read_table('News_info_validate_seg.txt',sep =  '\t' , header = None)
labels = train[2].tolist()

with open('tokenizer_with_unlabel.pk', 'rb') as fin:
    tokenizer = pickle.load(fin)
fin.close()

list_sentences_train = train[5].fillna('')
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
train_x = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen,truncating='pre',padding='pre')
train_y = to_categorical(labels)

def train_dl(mode,model,file_path,maxlen = 100,max_features = 30000,embed_size = 32,batch_size = 32,epochs = 5)
    if mode == 1:#01二分类
        train_labels = []
        train_texts = []
        for i in range(len(labels)):
            if labels[i] == 1 or labels[i] == 0 :
                train_labels.append(labels[i])
                train_texts.append(train[5][i])
        train_2 = pd.DataFrame({'text':train_texts})
        list_sentences_train = train_2['text'].fillna('')
        list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
        train_x = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen,truncating='pre',padding='pre')
        train_y = to_categorical(train_labels)
    if mode == 2:#01 2二分类
        _labels = []
        for i in range(len(labels)):
            if labels[i] == 2:
                _labels.append(1)
            else:
                _labels.append(0)        
        train_y = to_categorical(_labels)
    if mode == 3:#012三分类
        pass
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early] #early
    hist = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)


dims = [100,200,300,400,500,600,700,800,900,1000]

for i in range(1,3):
    if i == 3:
        layer = 3
    else:
        layer = 2
    for j in dims:
        train_dl(mode = i,model = textcnn_model(layer),embed_size = j,file_path="textcnn_model_mode"+str(i)+"dm"+str(j)+".hdf5")
        train_dl(mode = i,model = lstm_cnn_model(layer),embed_size = j,file_path="lstmcnn_model"+str(i)+"dm"+str(j)+"hdf5")    
        train_dl(mode = i,model =lstm_model(layer),embed_size = j,file_path="lstm_model"+str(i)+"dm"+str(j)+"hdf5")
        train_dl(mode = i,model = gru_model(layer),embed_size = j,file_path="gru_model"+str(i)+"dm"+str(j)+"hdf5")
