import numpy as np
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint
from models import ThreeLayerLSTM, ThreeLayerLSTMandCNN
from DataGenerator import DataGenerator36, DataGenerator18, DataGenerator36CNN
from data_org import data_org


EPOCHS = 200
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

data_org = data_org(dataset=DATASET)

if MODEL == '36IN3LSTMCNN':
    ThreeLayerLSTMCNN = ThreeLayerLSTMandCNN(L1=L1, L2=L2, L3=L3, t=TIMESTEP,
                                             num_classes=NUM_CLASSES,
                                             data_dim=36,
                                             imgHeight=IMG_HEIGHT,
                                             imgWidth=IMG_WIDTH)
    model = ThreeLayerLSTMCNN.build_network()

    # Generate training data from .npy file
    training_generator = DataGenerator36CNN(data_org.partition['Training'],
                                            data_org.labels,
                                            data_org.label_ids,
                                            batch_size=BATCH_SIZE, t=TIMESTEP,
                                            n_classes=NUM_CLASSES,
                                            imgHeight=IMG_HEIGHT,
                                            imgWidth=IMG_WIDTH)

    # Generate validation data from .npy file
    val_generator = DataGenerator36CNN(data_org.partition['Validation'],
                                       data_org.labels, data_org.label_ids,
                                       batch_size=BATCH_SIZE, t=TIMESTEP,
                                       n_classes=NUM_CLASSES,
                                       imgHeight=IMG_HEIGHT,
                                       imgWidth=IMG_WIDTH)

    # Training

    tensorboard = TensorBoard(
                        log_dir='logs/{}'.format(time()),
                        histogram_freq=2,
                        batch_size=BATCH_SIZE,
                        write_graph=True,
                        write_images=True)
    checkpoint = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}.hdf5".format(MODEL, EPOCHS, time()),
        monitor='val_acc',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard, checkpoint],
                        use_multiprocessing=False,
                        workers=WORKERS)

    model.save('trained_models/{}_model_{}_final.h5'.format(MODEL, time()))


if MODEL == '36IN3LSTM':

    ThreeLayerLSTM = ThreeLayerLSTM(L1=L1, L2=L2, L3=L3, t=TIMESTEP,
                                    num_classes=NUM_CLASSES, data_dim=36)
    model = ThreeLayerLSTM.build_network()

    # model = load_model(
    #     'trained_models/36IN3LSTM-500-2018-11-20 11:49:55.662198.hdf5')

    # Generate training data from .npy file
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
                        histogram_freq=2,
                        write_graph=True,
                        write_images=True)

    checkpoint = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}.hdf5".format(MODEL, EPOCHS, time()),
        monitor='val_acc',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard, checkpoint],
                        use_multiprocessing=False,
                        workers=WORKERS)

    model.save('trained_models/{}_model_{}_final.h5'.format(MODEL, time()))

    testing_generator = DataGenerator36(data_org.partition['Test'],
                                        data_org.labels, data_org.label_ids,
                                        batch_size=1, t=30,
                                        n_classes=10)
    total = 0
    correct = 0
    for i in range(1020):
        batch = testing_generator.__getitem__(i)
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
    total = 0
    correct = 0
    for i in range(1020):
        batch = training_generator.__getitem__(i)
        video = batch[0]
        label = batch[1][0]

        output = model.predict(np.array(video), batch_size=1)
        output = [np.round(elem, 1) for elem in output][0]

        max = np.argmax(output)
        total += 1
        if np.argmax(label) == max:
            correct += 1

    print('Training_Total: ', total)
    print('Training_Correct: ', correct)
    print('Training_Accuracy: ', correct/total)


if MODEL == '18IN3LSTM':
    ThreeLayerLSTM = ThreeLayerLSTM(L1=L1, L2=L2, L3=L3, t=TIMESTEP,
                                    num_classes=NUM_CLASSES, data_dim=18)
    model = ThreeLayerLSTM.build_network()

    # Generate training data from .npy file
    training_generator = DataGenerator18(data_org.partition['Training'],
                                         data_org.labels, data_org.label_ids,
                                         batch_size=BATCH_SIZE, t=TIMESTEP,
                                         n_classes=NUM_CLASSES)

    # Generate validation data from .npy file
    val_generator = DataGenerator18(data_org.partition['Validation'],
                                    data_org.labels, data_org.label_ids,
                                    batch_size=BATCH_SIZE, t=TIMESTEP,
                                    n_classes=NUM_CLASSES)

    # Training

    tensorboard = TensorBoard(
                        log_dir='logs/{}'.format(time()),
                        histogram_freq=2,
                        write_graph=True,
                        write_images=True)

    checkpoint = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}.hdf5".format(MODEL, EPOCHS, time()),
        monitor='val_acc',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard, checkpoint],
                        use_multiprocessing=True,
                        workers=WORKERS)

    model.save('trained_models/{}_model_{}_final.h5'.format(MODEL, time()))
