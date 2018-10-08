from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import time


class ThreeLayerLSTM():
    def __init__(self):
        self.L1 = 128
        self.L2 = 128
        self.L3 = 128

        self.data_dim = 18
        self.t = 1000
        self.num_classes = 8


    def build_network(self):
        model = Sequential()
        model.add(LSTM(self.L1, return_sequences=True,
                       input_shape=(self.t, self.data_dim),
                       activation='relu'))
        # model.add(Dropout(0.5))
        model.add(LSTM(self.L2, return_sequences=True, activation='relu'))
        # model.add(Dropout(0.5)
        model.add(LSTM(self.L3, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        return model
