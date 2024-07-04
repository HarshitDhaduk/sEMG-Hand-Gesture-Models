import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, LayerNormalization, Bidirectional, LSTM, DepthwiseConv2D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # Time-based local feature extraction using 1D Convolution
    x = Conv1D(filters=128, kernel_size=10, strides=6, activation='elu', padding='same')(inputs)
    x = LayerNormalization()(x)

    # Global feature extraction using FFT and 1D Convolution
    fft_layer = tf.signal.fft(tf.cast(inputs, tf.complex64))
    fft_layer = tf.math.real(fft_layer)
    y = Conv1D(filters=128, kernel_size=10, strides=6, activation='elu', padding='same')(fft_layer)
    y = LayerNormalization()(y)
    ifft_layer = tf.signal.ifft(tf.cast(y, tf.complex64))
    ifft_layer = tf.math.real(ifft_layer)

    # DepthWidth Separable Convolution for feature fusion
    combined = tf.concat([x, ifft_layer], axis=-1)
    z = DepthwiseConv2D(kernel_size=(3, 1), padding='same', activation='elu')(combined)
    z = Conv1D(filters=128, kernel_size=1, activation='elu')(z)
    z = LayerNormalization()(z)
    
    # BiLSTM for capturing temporal dependencies
    z = Bidirectional(LSTM(64, return_sequences=True))(z)
    z = GlobalAveragePooling1D()(z)
    
    # Flattening and fully connected layers for classification
    z = Flatten()(z)
    z = Dense(128, activation='elu')(z)
    z = LayerNormalization()(z)
    z = Dropout(0.2)(z)
    z = Dense(512, activation='elu')(z)
    z = LayerNormalization()(z)
    z = Dropout(0.2)(z)
    outputs = Dense(num_classes, activation='softmax')(z)

    model = Model(inputs, outputs)
    return model

# Parameters
input_shape = (time_steps, num_channels)
num_classes = 10  # Number of gesture classes

# Build and compile model
model = build_model(input_shape)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
