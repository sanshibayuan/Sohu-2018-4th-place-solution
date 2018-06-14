from keras.models import Model
from keras.layers import Dense, Embedding, Input,BatchNormalization
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout,Conv1D, CuDNNLSTM,GlobalAveragePooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, normalize
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
import pickle


def textcnn_model(layer):    
        filter_sizes = [1,2,3,5]	 #通过不同 filter_size 的 filter 获取不同宽度的视野
        num_filters =32 # 32*4=128
        inp = Input(shape=(maxlen, ))
        x = Embedding(max_features, embed_size)(inp)#通道（Channels）
        x = SpatialDropout1D(0.4)(x)
        x = Reshape((maxlen, embed_size, 1))(x)
        
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
                                                                                        activation='elu')(x)
        
        maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
        maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)        
        z = Concatenate(axis=1)([maxpool_0, maxpool_1,maxpool_2,maxpool_3])   
        z = Flatten()(z)
        z = Dropout(0.1)(z)
        outp = Dense(layer, activation="sigmoid")(z)
        
        model = Model(inputs=inp, outputs=outp)
        model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        return model

def lstm_model(layer):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(layer, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def lstm_cnn_model(layer):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True,dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    
    x = BatchNormalization()(x)
    x = Dense(50, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    outp = Dense(layer, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def gru_model(layer):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(layer, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def predict_dl(model,file_path):
    model.load_weights(file_path)
    y_test = model.predict(X_te)
    return y_test


    