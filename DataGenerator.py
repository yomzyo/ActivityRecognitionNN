import numpy as np
import keras
import os
import json


class DataGenerator(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_videos, labels, label_ids, batch_size=1,
                 dim=(1000, 18), n_channels=1, n_classes=8, shuffle=True):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.label_ids = label_ids
        self.list_videos = list_videos
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_videos) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_videos_temp = [self.list_videos[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_videos_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_videos))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

# Input is list of videos
    def __data_generation(self, list_videos_temp):
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        y = np.zeros((self.batch_size), dtype=int)
        # Generate data
        for i, video in enumerate(list_videos_temp):
            # Store sample

            for frame_index, frame in enumerate(sorted(
                                    os.listdir('datasets/UCF-101json/'
                                                + self.labels[video]
                                                + '/' + video))):
                json_file = 'datasets/UCF-101json/' + self.labels[video] \
                            + '/' + video + '/' + frame
                with open(json_file) as jf:
                    json_data = json.load(jf)
                    for point_index, point in enumerate(
                                            json_data['part_candidates'][0]):
                        value = _cantor(json_data['part_candidates'][0]
                                        [str(point)][0:2])[0]
                        value = value / 100000
                        X[i, frame_index, point_index, ] = value

            y[i] = self.label_ids[self.labels[video]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def _cantor(x):
    if not x:
        return [0]
    else:
        return [int((((x[0]) + x[1])*x[0] + x[1] + 1)/2 + x[1])]
