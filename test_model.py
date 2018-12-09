import numpy as np
from DataGenerator import DataGenerator36
from data_org import data_org
from keras.models import load_model

# Generate training data from .npy file

data_org = data_org(dataset='ntu_rgbd_dataset')

training_generator = DataGenerator36(data_org.partition['Training'],
                                     data_org.labels, data_org.label_ids,
                                     batch_size=1, t=30,
                                     n_classes=10)

correct = 0
total = 0

model = load_model(
    'comeplete/trained_models/36IN_LSTM500EPOCHS_V2/36IN3LSTM-500-1544224618.4687705.hdf5')
'''
for i in range(1020):
    batch = training_generator.__getitem__(i)
    video = batch[0]
    label = batch[1][0]

    output = model.predict(np.array(video), batch_size=1)
    output = [np.round(elem, 1) for elem in output][0]

    print('Label: ', label)

    for action, id in data_org.label_ids.items():
        if id == np.argmax(label):
            print(id, ' ', action)

    print('Output: ', output)
    max = np.argmax(output)
    for action, id in data_org.label_ids.items():
        if id == max:
            print(id, ' ', action)

    total += 1
    if np.argmax(label) == max:
        correct += 1

    print('\n')

print('Total: ', total)
print('Correct: ', correct)
print('Accuracy: ', correct/total)
'''

score = model.evaluate_generator(training_generator, steps=1020, verbose=1)
print(model.metrics_names)
print(score)
