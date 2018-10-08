import numpy as np
from time import time
from keras.callbacks import TensorBoard
from models import ThreeLayerLSTM
import os
from DataGenerator import DataGenerator

EPOCHS = 100            # TODO
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


ThreeLayerLSTM = ThreeLayerLSTM()
model = ThreeLayerLSTM.build_network()


# Generate training data from .npy file
training_generator = DataGenerator(partition['train'], labels, label_ids)
# Generate validation data from .npy file
val_generator = DataGenerator(partition['validation'], labels, label_ids)

# Training

tensorboard = TensorBoard(
                    log_dir='logs/{}'.format(time()),
                    histogram_freq=0,
                    write_graph=True,
                    write_images=True)

model.fit_generator(generator=training_generator,
                    validation_data=val_generator,
                    epochs=10,
                    callbacks=[tensorboard])
