from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


class ThreeLayerLSTM():
    def __init__(self, L1=500, L2=500, L3=500, t=1000,
                 num_classes=8, data_dim=18):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.data_dim = data_dim
        self.t = t
        self.num_classes = num_classes

    def build_network(self):
        model = Sequential()
        model.add(LSTM(self.L1, return_sequences=True,
                       input_shape=(self.t, self.data_dim),
                       activation='tanh'))
        # model.add(Dropout(0.5))
        model.add(LSTM(self.L2, return_sequences=True, activation='tanh'))
        # model.add(Dropout(0.5)
        model.add(LSTM(self.L3, activation='tanh'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        return model
