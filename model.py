import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers.experimental import preprocessing

def plot_loss(history):
    """ Plot model loss
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

def train_mnist(trainX, trainY, verbose=0):
    """ Train and checkpoint MLP for MNIST
    
    Args
     - trainX : flattened images
     - trainY : digit labels
    
    Returns
     - model   : trained model
     - history : training history
    """
    
    # Form model
    model = models.Sequential([
      layers.Dense(128,input_shape=(784,),activation='relu'),
      layers.Dense(10)
    ])

    # Compile model
    eta = 0.001
    model.compile(
        optimizer=tf.keras.optimizers.SGD(eta),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Callback to chheckpoint models
    checkpoint_callback = ModelCheckpoint(
        filepath="model/mnist/model{epoch}.hdf5",
        save_weights_only=True,
        save_best_only=False)

    # Train model
    history = model.fit(
        trainX, 
        trainY,
        validation_split=0.2,
        verbose=verbose,
        epochs=20,
        callbacks = checkpoint_callback
    )
    
    return(model, history)

def train_fuel(trainX, trainY, verbose=0):
    """ Train and checkpoint MLP for fuel efficiency data set
    
    Args
     - trainX : fuel efficiency features
     - trainY : miles per gallon (MPG) labels
    
    Returns
     - model   : trained model
     - history : training history
    """
    
    # Normalization
    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(trainX))

    # Callback to chheckpoint models
    checkpoint_callback = ModelCheckpoint(
        filepath="model/fuel/model{epoch}.hdf5",
        save_weights_only=True,
        save_best_only=False)
    
    # Form model
    model = keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # Compile model
    model.compile(loss=MeanSquaredError(),
                  optimizer=tf.keras.optimizers.SGD())

    # Train model
    history = model.fit(
        trainX, 
        trainY,
        validation_split=0.2,
        verbose=verbose, 
        epochs=20,
        callbacks = checkpoint_callback
    )
    
    return(model, history)