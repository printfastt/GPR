import keras
import tensorflow as tf
import toolkit as tk
tf.compat.v1.disable_eager_execution()

"""
Options:
    file:           loads model from the passed file
                    otherwise will load from current tr/te folder.
"""
def test(X_Data_test, **kwargs):
    imageSizeDp = X_Data_test.shape[1]
    imageSizeDT = X_Data_test.shape[3]
    X_Data_test = X_Data_test.reshape(X_Data_test.shape[0], imageSizeDp, 1, imageSizeDT, 1)
    
    if 'file' in kwargs:
        model = keras.models.load_model(kwargs['file'])
    else:
        model = keras.models.load_model(tk.getPath())
 
    predictions = model.predict(X_Data_test)
    predictions = predictions.flatten()

    return predictions
    
    
    
    

        

    