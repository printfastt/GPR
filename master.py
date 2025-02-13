import toolkit as tk
from testmodel import test
from trainmodel import train
"""
#train datasets: 1022cp, 0914cp, 0524cp, 1014cp, 1022hh, 0914hh, 0524hh, 1014hh, 914cp1022cp1014cp0524cp, 914hh1022hh1014hh0524hh, GaTechTe
#test datasets: 1022cp, 0914cp, 0524cp, 1014cp, 1022hh, 0914hh, 0524hh, 1014hh, GaTechTe
#models: 'resnet18', 'resnet38','pscnn_v1', 'resnetinspired'


NOTE: make github for this.
NOTE: make all analyzing, showing, saving functions available from master.
Note: plotOverlay() is not working correctly right now.

Improvements List:
    Add functionality for ROC log tpr,fpr,auc data storage & plot comparison.
    Add functionality for multiple train/test cycles, where the SeqConf's are averaged together.

"""

#Model Choice
model = 'pscnn_v1'

#Data Manipulation
train_datasets = ['gatechtr']
test_datasets = ['gatechte']
image_depth = (0,100)
#image_depth_list = [(25,100),(0,100),(0,200)]
negativewires = False


#Model Adjustment
activationFunction = 'relu'
loss = 'binary_crossentropy'
num_filters = 16
epoch_num = 10
kSize = 3
dropoutAmount = 0.4
batch_size = 64
optimizer = 'Adam'

#Run procedures
iterations = 1
savefolderpath = 'depth25to100test'                                                      
verbose = 1
#retrain = 1                                                                    #For multiple tests, should each train_dataset train for every test, or should it use the same one?

#Display
threshold = .99

#PPTX AutoCreation.
autocreate = True
pptxEnvPath = "C:/Users/carso.LAPTOP/anaconda3/envs/pptx_env/python.exe"

if __name__ == "__main__":
    for train_dataset in train_datasets:
        for test_dataset in test_datasets:
                            
                tk.createConfigFile(model, train_dataset, test_dataset, savefolderpath, threshold)
    
                
    #init Hyperparameters and return as dict
                Hyperparameters = tk.initHyperparameters(
                                                         model,
                                                         activationFunction, 
                                                         num_filters, 
                                                         epoch_num, 
                                                         kSize, 
                                                         dropoutAmount, 
                                                         batch_size, 
                                                         optimizer, 
                                                         loss, 
                                                         iterations,
                                                         verbose
                                                         )
                
    #init Datasets.
                X_Data_train, Y_Data_train, X_Data_test, Y_Data_test, dncr, DataSpecs = tk.initDatasets(
                                                                                                        traindataset=train_dataset, 
                                                                                                        testdataset=test_dataset, 
                                                                                                        image_depth=image_depth, 
                                                                                                        negativewires=negativewires, 
                                                                                                        Hyperparameters=Hyperparameters
                                                                                                        )
                            
                
                
                
                
            
    #Train/Test                 
                train(X_Data_train, Y_Data_train, model, Hyperparameters)
                predictions = test(X_Data_test)
                tk.savePredictions(predictions)
                
    
    
    #Data Display
                tk.plotPredictionsAuto(predictions)
                tk.plotROCAuto(Y_Data_test, predictions, dncr)
                #tk.plotROCLogAuto(Y_Data_test, predictions, dncr)
                #tk.plotConfusionMatrixAuto(Y_Data_test, predictions, dncr)
                #tk.plotOverlayAuto(Y_Data_test, predictions)
              
                
              
    #Create PPTX. ONLY IF 'pptx_env' IS SET UP IN ANACONDA. conda create env --
    #Must change file location in makePresAuto.py, and pptx save name.
                #tk.createPPTX(autocreate, pptxEnvPath)



"""
if __name__ == "__main__":
    for image_depth in image_depth_list:
        if image_depth == (25,100):
            savefolderpath = '25to100'
        elif image_depth == (0,100):
            savefolderpath = '0to100'
        elif image_depth == (0,200):
            savefolderpath = '0to200'
        for train_dataset in train_datasets:
            for test_dataset in test_datasets:
                
                
                tk.createConfigFile(model, train_dataset, test_dataset, savefolderpath, threshold)
    
                
    #init Hyperparameters and return as dict
                Hyperparameters = tk.initHyperparameters(
                                                         model,
                                                         activationFunction, 
                                                         num_filters, 
                                                         epoch_num, 
                                                         kSize, 
                                                         dropoutAmount, 
                                                         batch_size, 
                                                         optimizer, 
                                                         loss, 
                                                         iterations,
                                                         verbose
                                                         )
                
    #init Datasets.
                X_Data_train, Y_Data_train, X_Data_test, Y_Data_test, dncr, DataSpecs = tk.initDatasets(
                                                                                                        traindataset=train_dataset, 
                                                                                                        testdataset=test_dataset, 
                                                                                                        image_depth=image_depth, 
                                                                                                        negativewires=negativewires, 
                                                                                                        Hyperparameters=Hyperparameters
                                                                                                        )
                            
                
                
                
                
            
    #Train/Test                 
                train(X_Data_train, Y_Data_train, model, Hyperparameters)
                predictions = test(X_Data_test)
                tk.savePredictions(predictions)
                
    
    
    #Data Display
                tk.plotPredictionsAuto(predictions)
                tk.plotROCAuto(Y_Data_test, predictions, dncr)
                #tk.plotROCLogAuto(Y_Data_test, predictions, dncr)
                #tk.plotConfusionMatrixAuto(Y_Data_test, predictions, dncr)
                #tk.plotOverlayAuto(Y_Data_test, predictions)
              
                
              
    #Create PPTX. ONLY IF 'pptx_env' IS SET UP IN ANACONDA. conda create env --
    #Must change file location in makePresAuto.py, and pptx save name.
                #tk.createPPTX(autocreate, pptxEnvPath)

  
                
    
        """        
            
            
            
            
            
            

