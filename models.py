from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import concatenate
from keras import optimizers


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
                       activation='tanh',
                       dropout=.4))
        model.add(LSTM(self.L2, return_sequences=True, activation='tanh',
                       dropout=.4))
        model.add(LSTM(self.L3, activation='tanh', dropout=.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])

        return model


class ThreeLayerLSTMandCNN():
    def __init__(self, L1=500, L2=500, L3=500, t=1000,
                 num_classes=10, data_dim=36, imgHeight=540, imgWidth=960):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.data_dim = data_dim
        self.t = t
        self.num_classes = num_classes
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

    def build_network(self):
        # Build 3 LSTM layers for temporal recognition
        temporal_model = Sequential()
        temporal_model.add(LSTM(self.L1, return_sequences=True,
                           input_shape=(self.t, self.data_dim),
                           activation='tanh', dropout=.4))
        # model.add(Dropout(0.5))
        temporal_model.add(LSTM(self.L2, return_sequences=True,
                                activation='tanh', dropout=.4))
        # model.add(Dropout(0.5)
        temporal_model.add(LSTM(self.L3, activation='tanh', dropout=.4))
        temporal_model.add(Dense(100))

        # Skeleton sequence output

        temporal_input = Input(shape=(self.t, self.data_dim))
        encoded_temp = temporal_model(temporal_input)

        # build spatial network
        spat_model = Sequential()
        spat_model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                       input_shape=(self.imgHeight, self.imgWidth, 3)))
        spat_model.add(Conv2D(64, (3, 3), activation='relu'))
        spat_model.add(MaxPooling2D(2, 2))
        spat_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        spat_model.add(Conv2D(128, (3, 3), activation='relu'))
        spat_model.add(MaxPooling2D(2, 2))
        spat_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        spat_model.add(Conv2D(256, (3, 3), activation='relu'))
        spat_model.add(Conv2D(256, (3, 3), activation='relu'))
        spat_model.add(MaxPooling2D(2, 2))
        spat_model.add(Flatten())
        spat_model.add(Dense(100))

        spat_input = Input(shape=(self.imgHeight, self.imgWidth, 3))
        encoded_spat = spat_model(spat_input)

        merged = concatenate([encoded_temp, encoded_spat])

        output = Dense(self.num_classes, activation='softmax')(merged)
        model = Model(inputs=[temporal_input, spat_input], outputs=output)
        adam = optimizers.adam(lr=0.00001)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        return model
