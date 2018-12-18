import numpy as np
import keras
import os
import json
import cv2


class DataGenerator18(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_videos, labels, label_ids, batch_size=1,
                 t=1000, n_classes=10, shuffle=True):
        # Initialization
        self.t = t
        self.batch_size = batch_size
        self.labels = labels
        self.label_ids = label_ids
        self.list_videos = list_videos
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
        X = np.zeros((self.batch_size, self.t, 18))
        y = np.zeros((self.batch_size), dtype=int)
        # Generate data
        for i, video in enumerate(list_videos_temp):
            # Store sample
#            for frame_index, frame in enumerate(sorted(
#                                    os.listdir('datasets/UCF-101json/'
#                                                + self.labels[video]
#                                                + '/' + video))):
#                json_file = 'datasets/UCF-101json/' + self.labels[video] \
#                            + '/' + video + '/' + frame
#                with open(json_file) as jf:
#                    json_data = json.load(jf)
            for frame_index, frame in enumerate(sorted(os.listdir(
                        'datasets/ntu_rgb_dataset_PREP_REAL_TIME/' + video))):
                if '.json' in frame:
                    json_file = 'datasets/ntu_rgb_dataset_PREP_REAL_TIME/'\
                                + video + '/' + frame
                    with open(json_file) as jf:
                        json_data = json.load(jf)
                        for point_index, point in enumerate(
                                            json_data['part_candidates'][0]):
                            value = _cantor(x=json_data['part_candidates']
                                            [0][str(point)][0:2])[0]
                            value = value / 2073600
                            X[i, frame_index, point_index, ] = value

            y[i] = self.label_ids[self.labels[video]]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def _cantor(x):
    if not x:
        return [0]
    else:
        return [int((((x[0]) + x[1])*x[0] + x[1] + 1)/2 + x[1])]


class DataGenerator36(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_videos, labels, label_ids, batch_size=1,
                 t=1000, n_classes=10, shuffle=True):
        # Initialization
        self.t = t
        self.batch_size = batch_size
        self.labels = labels
        self.label_ids = label_ids
        self.list_videos = list_videos
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
        X = np.zeros((self.batch_size, self.t, 36))
        y = np.zeros((self.batch_size), dtype=int)
        # Generate data
        for i, video in enumerate(list_videos_temp):
            # Store sample
            for frame_index, frame in enumerate(sorted(os.listdir(
                        'datasets/ntu_rgb_dataset_PREP_REAL_TIME/' + video))):
                if '.json' in frame:
                    json_file = 'datasets/ntu_rgb_dataset_PREP_REAL_TIME/'\
                                + video + '/' + frame
                    with open(json_file) as jf:
                        json_dta = json.load(jf)
                        for point_index, point in enumerate(
                                            json_dta['part_candidates'][0]):
                            value_temp =\
                                json_dta['part_candidates'][0][str(point)][0:2]
                            if not value_temp:
                                value_temp = [0, 0]

                            value = [x / 1920 for x in value_temp]
                            X[i, frame_index, point_index*2, ] \
                                = value[0]
                            X[i, frame_index, point_index*2+1, ] \
                                = value[1]
            y[i] = self.label_ids[self.labels[video]]

        print(y)
        print(keras.utils.to_categorical(y, num_classes=self.n_classes))
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGenerator36CNN(keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, list_videos, labels, label_ids, batch_size=1,
                 t=1000, n_classes=8, shuffle=True, imgHeight=1080,
                 imgWidth=1920):
        # Initialization
        self.t = t
        self.batch_size = batch_size
        self.labels = labels
        self.label_ids = label_ids
        self.list_videos = list_videos
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight
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

        spatial = np.zeros((self.batch_size, self.imgHeight, self.imgWidth, 3))
        temporal = np.zeros((self.batch_size, self.t, 36))
        y = np.zeros((self.batch_size), dtype=int)
        # Generate data
        for i, video in enumerate(list_videos_temp):
            # Store sample

            for frame_index, frame in enumerate(sorted(os.listdir(
                        'datasets/ntu_rgb_dataset_PREP_REAL_TIME/' + video))):
                if '.json' in frame:
                    json_file = 'datasets/ntu_rgb_dataset_PREP_REAL_TIME/'\
                                + video + '/' + frame
                    with open(json_file) as jf:
                        json_dta = json.load(jf)
                        for point_index, point in enumerate(
                                            json_dta['part_candidates'][0]):
                            value_temp =\
                                json_dta['part_candidates'][0][str(point)][0:2]
                            if not value_temp:
                                value_temp = [0, 0]

                            value = [x / 1920 for x in value_temp]
                            temporal[i, frame_index, point_index*2, ] \
                                = value[0]
                            temporal[i, frame_index, point_index*2+1, ] \
                                = value[1]

            image_path = 'datasets/ntu_rgb_dataset_PREP_REAL_TIME/' + video +\
                '/' + video + '.jpg'
            if os.path.exists(image_path):
                img = cv2.resize(cv2.imread(image_path, 1), (0, 0),
                                 fx=0.25, fy=0.25)
                temp_img = [[[float(v)/255 for v in c] for c in r] for r in img]
                spatial[i, ] = temp_img
            y[i] = self.label_ids[self.labels[video]]

        return [temporal, spatial], keras.utils.to_categorical(
                                                y, num_classes=self.n_classes)
