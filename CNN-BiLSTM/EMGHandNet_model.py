import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, TimeDistributed, Flatten
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split

def create_emg_handnet_model(input_shape, num_classes):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=9, strides=1, 
                                     padding='same', kernel_initializer=HeNormal(), 
                                     kernel_regularizer=tf.keras.regularizers.l2(10**-4)), 
                              input_shape=input_shape))
    model.add(TimeDistributed(BatchNormalization(momentum=0.95, epsilon=10**-6)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2093)))

    # Max Pooling Layer
    model.add(TimeDistributed(MaxPooling1D(pool_size=2, strides=1, padding='same')))

    # 2nd Convolutional Layer
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, strides=1, 
                                     padding='same', kernel_initializer=HeNormal(), 
                                     kernel_regularizer=tf.keras.regularizers.l2(10**-4))))
    model.add(TimeDistributed(BatchNormalization(momentum=0.95, epsilon=10**-6)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2093)))

    # 3rd Convolutional Layer
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5, strides=1, 
                                     padding='same', kernel_initializer=HeNormal(), 
                                     kernel_regularizer=tf.keras.regularizers.l2(10**-4))))
    model.add(TimeDistributed(BatchNormalization(momentum=0.95, epsilon=10**-6)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2093)))

    # 4th Convolutional Layer
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, strides=1, 
                                     padding='same', kernel_initializer=HeNormal(), 
                                     kernel_regularizer=tf.keras.regularizers.l2(10**-4))))
    model.add(TimeDistributed(BatchNormalization(momentum=0.95, epsilon=10**-6)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.2093)))

    # Flatten the output
    model.add(TimeDistributed(Flatten()))

    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Bidirectional(LSTM(200)))

    # Dense Layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2093))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Define input shape and number of classes
input_shape = (25, 20, 10)  # SL=25, TS=20, NC=10 for NinaPro DB1
num_classes = 52  # Number of classes for NinaPro DB1

# Create the model
model = create_emg_handnet_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=10**-3, beta_1=0.9, beta_2=0.999),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load preprocessed data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Convert DataFrames to numpy arrays and reshape to original 3D shape
train_data = train_df.values.reshape((-1, 25, 20, 10))
test_data = test_df.values.reshape((-1, 25, 20, 10))

train_labels = train_labels_df.values
test_labels = test_labels_df.values


# One-hot encode the labels
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Train the model
model.fit(train_data, train_labels, epochs=200, batch_size=16, validation_data=(test_data, test_labels))

# Save the trained model
model.save('emg_handnet_model.h5')
