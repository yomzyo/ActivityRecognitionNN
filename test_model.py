import numpy as np
from time import time
from keras.callbacks import TensorBoard
from keras.models import load_model
from models import ThreeLayerLSTM
import os
from DataGenerator import DataGenerator18, DataGenerator36

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

model = load_model('2nd_trained_model.h5')

# Generate training data from .npy file
testing_generator = DataGenerator36(partition['test'], labels, label_ids)

model.evaluate_generator(testing_generator,
                         workers=16,
                         use_multiprocessing=True)
