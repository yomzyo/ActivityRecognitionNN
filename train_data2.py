import numpy as np
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint
from DataGenerator import DataGenerator36
from data_org import data_org
import types
import tempfile
import keras.models
# from make_keras_picklable import make_keras_picklable
import keras
import pickle

make_keras_picklable()

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


EPOCHS = 10
NUM_CLASSES = 10
BATCH_SIZE = 32
TIMESTEP = 30
DATASET = 'ntu_rgbd_dataset'
MODEL = '36IN3LSTM'
L1 = 500
L2 = 500
L3 = 500
WORKERS = 16
IMG_HEIGHT = 270
IMG_WIDTH = 480

# make_keras_picklable()
data_org = data_org(dataset=DATASET)


if MODEL == '36IN3LSTM':
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(L1, return_sequences=True,
              input_shape=(TIMESTEP, 36),
              activation='tanh',
              dropout=.4))
    model.add(keras.layers.LSTM(L2, return_sequences=True, activation='tanh',
              dropout=.4))
    model.add(keras.layers.LSTM(L3, activation='tanh', dropout=.4))
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    training_generator = DataGenerator36(data_org.partition['Training'],
                                         data_org.labels, data_org.label_ids,
                                         batch_size=BATCH_SIZE, t=TIMESTEP,
                                         n_classes=NUM_CLASSES)

    # Generate validation data from .npy file
    val_generator = DataGenerator36(data_org.partition['Validation'],
                                    data_org.labels, data_org.label_ids,
                                    batch_size=BATCH_SIZE, t=TIMESTEP,
                                    n_classes=NUM_CLASSES)

    # Training

    tensorboard = TensorBoard(
                        log_dir='logs/{}'.format(time()),
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)

    checkpoint_acc = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}_maxACC.hdf5".format(MODEL, EPOCHS, time()),
        monitor='val_acc',
        save_best_only=True)
    checkpoint_loss = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}._minLosshdf5".format(MODEL, EPOCHS, time()),
        monitor='val_loss',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard],
                        # callbacks=[tensorboard, checkpoint_acc, checkpoint_loss],
                        use_multiprocessing=False,
                        workers=WORKERS)

    testing_generator = DataGenerator36(data_org.partition['Test'],
                                        data_org.labels, data_org.label_ids,
                                        batch_size=1, t=30,
                                        n_classes=10)

    pickle.dumps(model)
    total = 0
    correct = 0
    for i in range(1020):
        batch = testing_generator.__getitem__(i)
        video = batch[0]
        label = batch[1][0]

        output = model.predict(np.array(video), batch_size=1)
        output = [np.round(elem, 1) for elem in output][0]

        max = np.argmax(output)
        total += 1
        if np.argmax(label) == max:
            correct += 1

    print('Total: ', total)
    print('Correct: ', correct)
    print('Accuracy: ', correct/total)
    model.save('trained_models/{}_model_{}_final.h5'.format(MODEL, time()))
