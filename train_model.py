import numpy as np
from time import time
import os
from keras.callbacks import TensorBoard
from models import ThreeLayerLSTM

EPOCHS = 100            # TODO
# NUM_CLASSES = 15
BATCH_SIZE = 128        # TODO
# CHUNK_SIZE = 28

ThreeLayerLSTM = ThreeLayerLSTM()
model = ThreeLayerLSTM.build_network()

x_data = np.load('training_data-1.npy')
y_data = np.load('training_data_labels-1.npy')


def _unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


x_data, y_data = _unison_shuffled_copies(x_data, y_data)

# Generate training data from .npy file
x_train = x_data[0:1000]
y_train = y_data[0:1000]
# Generate validation data from .npy file
x_validate = x_data[1000:1106]
y_validate = y_data[1000:1106]
# Training

tensorboard = TensorBoard(
                    log_dir='logs/{}'.format(time()),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True)

model.fit(x_train, y_train, batch_size=1000, epochs=EPOCHS,
          shuffle=False, validation_data=(x_validate, y_validate),
          callbacks=[tensorboard])
