import numpy as np
from time import time
from keras.callbacks import TensorBoard
from models import ThreeLayerLSTM
import os
from DataGenerator import DataGenerator18, DataGenerator36

EPOCHS = 10
# NUM_CLASSES = 15
dataset_folder = 'Dataset_Separation/UCF-101json'


partition = {}
labels = {}
label_ids = {}
for type in os.listdir(dataset_folder):
    partition[type] = []
    for i, activity in enumerate(os.listdir(dataset_folder + '/' + type)):
        label_ids[activity] = i
        for video in os.listdir(dataset_folder + '/' + type + '/' + activity):
            partition[type].append(video)
            labels[video] = activity


ThreeLayerLSTM = ThreeLayerLSTM(L1=64, L2=64, L3=64, t=1000,
                                num_classes=8, data_dim=36)
model = ThreeLayerLSTM.build_network()


# Generate training data from .npy file
training_generator = DataGenerator36(partition['train'], labels, label_ids)
# Generate validation data from .npy file
val_generator = DataGenerator36(partition['validation'], labels, label_ids)

# Training

tensorboard = TensorBoard(
                    log_dir='logs/{}'.format(time()),
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)

model.fit_generator(generator=training_generator,
                    validation_data=val_generator,
                    epochs=EPOCHS,
                    callbacks=[tensorboard],
                    use_multiprocessing=True,
                    workers=1)

model.save('trained_model.h5')
