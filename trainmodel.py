import toolkit as tk
from sklearn.utils import shuffle
from keras.layers import Dense , Flatten , Dropout 
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras import backend as K

def train(X_Data_train, Y_Data_train, model, Hyperparameters, **kwargs):
    if model == 'pscnn_v1':
        return train_pscnnv1(X_Data_train, Y_Data_train, Hyperparameters)
    if model == 'resnet18':
        pass
    if model == 'resnet34':
        pass
    
    

def train_pscnnv1(X_Data_train, Y_Data_train, Hyperparameters, **kwargs):
    imageSizeDp = X_Data_train.shape[1]
    imageSizeDT = X_Data_train.shape[3]
    
        
    X_Data_train = X_Data_train.reshape(X_Data_train.shape[0],X_Data_train.shape[1],X_Data_train.shape[2],X_Data_train.shape[3],1)
    X_Data_train = X_Data_train[:,0:imageSizeDp,:,:]

    X_train, y_train = shuffle(X_Data_train, Y_Data_train, random_state=42)

    sample_shape = (imageSizeDp, 1, imageSizeDT, 1)
    
    
    model1 = Sequential()
    model1.add(Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernalSize'], padding = "same", activation=Hyperparameters['activationFunction'], kernel_initializer='he_uniform', input_shape=sample_shape))
    model1.add(MaxPooling3D(pool_size=(2, 1, 2)))
    model1.add(BatchNormalization())
    
    model1.add(Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernalSize'], padding = "same", activation=Hyperparameters['activationFunction'], kernel_initializer='he_uniform', input_shape=sample_shape))
    model1.add(MaxPooling3D(pool_size=(2, 1, 2)))
    model1.add(BatchNormalization())


    model1.add(Conv3D(Hyperparameters['num_filters'], kernel_size=Hyperparameters['kernalSize'], padding = "same", activation=Hyperparameters['activationFunction'], kernel_initializer='he_uniform', input_shape=sample_shape))
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

        
    
def train_resnet18():
    pass

def train_resnet34():
    pass


    

