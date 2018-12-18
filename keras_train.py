
# -*- coding: utf-8 -*
from vectorize_data import Preprocess, train_val_split, load
from load_data import load_data
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.metrics import top_k_categorical_accuracy

"""
top5 = 0.77
top1 = 0.45

"""

if __name__ == '__main__':
    """
    for first time usage:
    first run vectorize_data.py, then run keras_train.py
    otherwise:
    run keras_train.py
    
    notes:
    n of vocabulary in corpus = 114260, keeping only thousands lose tremendous amount of information
    n of class: 1429; top5 of 66% is acceptable
    """

    import time
    start = time.time()

    x,y = load(path='np_processed_data')

    x_train, y_train, x_val, y_val = train_val_split(x, y, 0.8, shuffle_seed=1322) # baseline seed: 22

    print('verify seed by logging y_val[34]class:{} '.format(np.argmax(y_val[34])))  # 907
    print x_train.shape
    print y_train.shape
    print x_val.shape
    print y_val.shape

    model = Sequential()

    model.add(Dense(1024, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dense(1024, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dense(y_train.shape[1], activation='softmax'))

    top_k_metric = keras.metrics.top_k_categorical_accuracy
    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=[top_k_metric,'accuracy'])


    model.fit(x_train, y_train,
              validation_data=(x_val, y_val),
              epochs=200,
              batch_size=128,
              verbose = 2,
              callbacks=[EarlyStopping(patience=3), checkpointer, TensorBoard()])


    end = time.time()
    print('execution time in {} minutes'.format((end-start)/60))

