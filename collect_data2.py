# This script will convert datasets to numpy file wich later will be used for

import numpy as np
import os
import json

starting_value = 1

while True:
    file_name = 'training_data-{}.npy'.format(starting_value)
    file2_name = 'training_data_labels-{}.npy'.format(starting_value)

    if os.path.isfile(file_name) or os.path.isfile(file2_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)
        break


def main(file_name, file2_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    data_folder = 'datasets/UCF-101json'
    training_data = np.zeros((1106, 1000, 18), dtype=int)
    labels = np.zeros((1106, 8), dtype=int)
    print('Starting processing JSON files to {}'.format(file_name))

    activity_index = 0
    video_index = 0
    for activity_dir in os.listdir(data_folder):
        print('Processing ' + activity_dir + '...')
        for video in os.listdir(data_folder + '/' + activity_dir):
            frame_index = 0
            labels[video_index][activity_index] = 1
            for frame in os.listdir(
                        data_folder + '/' + activity_dir + '/' + video):
                json_file = data_folder + '/' + activity_dir  \
                            + '/' + video + '/' + frame
                with open(json_file) as jf:
                    json_data = json.load(jf)
                    point_index = 0
                    for point in json_data['part_candidates'][0]:
                        training_data[video_index][frame_index][point_index] \
                                = _cantor(json_data['part_candidates'][0]
                                          [str(point)][0:2])[0]

                        point_index += 1
                frame_index += 1
            video_index += 1
        activity_index += 1
        print('Activity ' + activity_dir + ' completed.')

    np.save(file_name, training_data)
    np.save(file2_name, labels)
    print('SAVED')

    training_data = []
    labels = []
    starting_value += 1
    file_name = 'training_data-{}.npy'.format(starting_value)
    file2_name = 'training_data_labels-{}.npy'.format(starting_value)


def _cantor(x):
    if not x:
        return [0]
    else:
        return [int((((x[0]) + x[1])*x[0] + x[1] + 1)/2 + x[1])]


main(file_name, file2_name, starting_value)
