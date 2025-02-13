import toolkit as tk
from sklearn.utils import shuffle
from keras.layers import Dense , Flatten , Dropout 
from keras.models import Sequential
from keras.layers import Input, Add, Conv3D, MaxPooling3D, BatchNormalization, Activation, GlobalAveragePooling3D
from keras.models import Model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras import backend as K

def train(X_Data_train, Y_Data_train, model, Hyperparameters, **kwargs):
        
    
    if model.lower() == 'pscnn_v1':
        return train_pscnnv1(X_Data_train, Y_Data_train, Hyperparameters)
    if model.lower() == 'resnet18':
        pass
    if model.lower() == 'resnet34':
        pass
    if model.lower() == 'resnetinspired':
        return train_ResNetInspired(X_Data_train, Y_Data_train, Hyperparameters)
    
    

def train_pscnnv1(X_Data_train, Y_Data_train, Hyperparameters, **kwargs):
    imageSizeDp = X_Data_train.shape[1]
    imageSizeDT = X_Data_train.shape[3]
    
        
    X_Data_train = X_Data_train.reshape(X_Data_train.shape[0],X_Data_train.shape[1],X_Data_train.shape[2],X_Data_train.shape[3],1)
    X_Data_train = X_Data_train[:,0:imageSizeDp,:,:]

    X_train, y_train = shuffle(X_Data_train, Y_Data_train, random_state=42)

    sample_shape = (imageSizeDp, 1, imageSizeDT, 1)
    
    
    model1 = Sequential()
    model1.add(Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernelSize'], padding = "same", activation=Hyperparameters['activationFunction'], kernel_initializer='he_uniform', input_shape=sample_shape))
    model1.add(MaxPooling3D(pool_size=(2, 1, 2)))
    model1.add(BatchNormalization())
    
    model1.add(Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernelSize'], padding = "same", activation=Hyperparameters['activationFunction'], kernel_initializer='he_uniform', input_shape=sample_shape))
    model1.add(MaxPooling3D(pool_size=(2, 1, 2)))
    model1.add(BatchNormalization())


    model1.add(Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernelSize'], padding = "same", activation=Hyperparameters['activationFunction'], kernel_initializer='he_uniform', input_shape=sample_shape))
    model1.add(MaxPooling3D(pool_size=(2, 1, 2)))
    model1.add(BatchNormalization())
    
    model1.add(Dropout(Hyperparameters['dropoutAmount']))
    model1.add(Flatten())
    model1.add(Dense(Hyperparameters['num_filters']*2,activation=Hyperparameters['activationFunction']))
    model1.add(Dense(1, activation="sigmoid"))
    
    model1.summary()
    model1.compile(optimizer=Hyperparameters['optimizer'], loss=Hyperparameters['loss'], metrics=['accuracy'])
    #K.set_value(model1.optimizer.learning_rate, Hyperparameters['learning_rate'])
    
    
    model1.fit(X_train, y_train, batch_size=Hyperparameters['batch_size'], epochs=Hyperparameters['epoch_num'],verbose = Hyperparameters['verbose'])
    
    model1.save(tk.getPath())
    
    K.clear_session()
    
    """
    good resnet hyperparam setup:
        #Model Adjustment
        activationFunction = 'relu'
        loss = 'binary_crossentropy'
        num_filters = 16
        epoch_num = 10
        kSize = 3
        dropoutAmount = 0.4
        batch_size = 64
        optimizer = 'Adam'
    """


def resnet_block(x, filters, kernel_size):
    """Residual block with two Conv3D layers and a skip connection. 
       Applies 1x1 convolution if needed for shortcut adjustment.
    """
    shortcut = x  # Save original input for the residual path
    
    # First convolutional layer
    x = Conv3D(filters, kernel_size=kernel_size, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second convolutional layer
    x = Conv3D(filters, kernel_size=kernel_size, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)

    # Adjust shortcut dimension if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv3D(filters, kernel_size=(1,1,1), padding="same", kernel_initializer="he_uniform")(shortcut)

    # Skip connection
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    
    return x

def train_ResNetInspired(X_Data_train, Y_Data_train, Hyperparameters, **kwargs):
    imageSizeDp = X_Data_train.shape[1]
    imageSizeDT = X_Data_train.shape[3]

    X_Data_train = X_Data_train.reshape(X_Data_train.shape[0], imageSizeDp, 1, imageSizeDT, 1)
    X_Data_train = X_Data_train[:, 0:imageSizeDp, :, :]

    X_train, y_train = shuffle(X_Data_train, Y_Data_train, random_state=42)
    
    sample_shape = (imageSizeDp, 1, imageSizeDT, 1)
    inputs = Input(shape=sample_shape)

    # Initial Conv Layer
    x = Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernelSize'], padding="same",
               kernel_initializer="he_uniform")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Residual Blocks
    x = resnet_block(x, Hyperparameters['num_filters'], Hyperparameters['kernelSize'])
    x = MaxPooling3D(pool_size=(2, 1, 2))(x)

    x = resnet_block(x, Hyperparameters['num_filters'] * 2, Hyperparameters['kernelSize'])
    x = MaxPooling3D(pool_size=(2, 1, 2))(x)

    x = resnet_block(x, Hyperparameters['num_filters'] * 4, Hyperparameters['kernelSize'])
    
    # Global Average Pooling for better feature aggregation
    x = GlobalAveragePooling3D()(x)

    # Fully Connected Layers
    x = Dense(Hyperparameters['num_filters'] * 2, activation=Hyperparameters['activationFunction'])(x)
    x = Dropout(Hyperparameters['dropoutAmount'])(x)
    outputs = Dense(1, activation="sigmoid")(x)

    # Build and compile model
    model = Model(inputs, outputs)
    model.summary()
    model.compile(optimizer=Hyperparameters['optimizer'], loss=Hyperparameters['loss'], metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=Hyperparameters['batch_size'], epochs=Hyperparameters['epoch_num'],
              verbose=Hyperparameters['verbose'])

    model.save(tk.getPath())

    K.clear_session()
    
    """
    activationFunction = 'relu'  # Keep this, as ReLU is ideal for deep CNNs
    loss = 'binary_crossentropy'  # Keep this, as it's a binary classification problem
    num_filters = 32  # Increase from 16 to 32 for better feature extraction
    epoch_num = 15  # Increase from 3 to 15 for better convergence
    kSize = 3  # Keep this, as it's a standard kernel size for feature extraction
    dropoutAmount = 0.3  # Reduce slightly to prevent excessive regularization
    batch_size = 128  # Increase from 64 to 128 for faster training and more stable batch norm
    optimizer = 'Adam'  # Keep this, as it's an efficient optimizer
    #learning_rate = 0.0005  # Manually set LR for more stable training
    """

        
    
def train_resnet18():
    pass

def train_resnet34():
    pass


    

