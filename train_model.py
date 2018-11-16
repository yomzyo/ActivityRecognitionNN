import numpy as np
import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from models import ThreeLayerLSTM, ThreeLayerLSTMandCNN
from DataGenerator import DataGenerator36, DataGenerator18, DataGenerator36CNN
from data_org import data_org

EPOCHS = 100
NUM_CLASSES = 10
BATCH_SIZE = 100
TIMESTEP = 30
DATASET = 'ntu_rgbd_dataset'
MODEL = '36IN3LSTMCNN'
L1 = 500
L2 = 500
L3 = 500
WORKERS = 1
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
                        log_dir='logs/{}'.format(datetime.datetime.now()),
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)
    checkpoint = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}.hdf5".format(
                                                    MODEL,
                                                    EPOCHS,
                                                    datetime.datetime.now()),
        monitor='val_acc',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard, checkpoint],
                        use_multiprocessing=False,
                        workers=WORKERS)

    model.save('trained_models/{}_model_{}_final.h5'.format(
                                                    MODEL,
                                                    datetime.datetime.now()))


if MODEL == '36IN3LSTM':
    ThreeLayerLSTM = ThreeLayerLSTM(L1=L1, L2=L2, L3=L3, t=TIMESTEP,
                                    num_classes=NUM_CLASSES, data_dim=36)
    model = ThreeLayerLSTM.build_network()

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
                        log_dir='logs/{}'.format(datetime.datetime.now()),
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)

    checkpoint = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}.hdf5".format(
                                                    MODEL,
                                                    EPOCHS,
                                                    datetime.datetime.now()),
        monitor='val_acc',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard, checkpoint],
                        use_multiprocessing=True,
                        workers=WORKERS)

    model.save('trained_models/{}_model_{}_final.h5'.format(
                                                    MODEL,
                                                    datetime.datetime.now()))

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
                        log_dir='logs/{}'.format(datetime.datetime.now()),
                        histogram_freq=0,
                        write_graph=True,
                        write_images=True)

    checkpoint = ModelCheckpoint(
        filepath="trained_models/{}-{}-{}.hdf5".format(
                                                    MODEL,
                                                    EPOCHS,
                                                    datetime.datetime.now()),
        monitor='val_acc',
        save_best_only=True)

    model.fit_generator(generator=training_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS,
                        callbacks=[tensorboard, checkpoint],
                        use_multiprocessing=True,
                        workers=WORKERS)

    model.save('trained_models/{}_model_{}_final.h5'.format(
                                                    MODEL,
                                                    datetime.datetime.now()))
