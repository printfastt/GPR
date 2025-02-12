import toolkit as tk
from testmodel import test
from trainmodel import train
"""
#train datasets: 1022cp, 0914cp, 0524cp, 1014cp, 1022hh, 0914hh, 0524hh, 1014hh, 914cp1022cp1014cp0524cp, 914hh1022hh1014hh0524hh, GaTechTe
#test datasets: 1022cp, 0914cp, 0524cp, 1014cp, 1022hh, 0914hh, 0524hh, 1014hh, GaTechTe
#models: 'resnet18', 'resnet38','pscnn_v1'


NOTE: make github for this.
NOTE: make all analyzing, showing, saving functions available from master.

Improvements List:
    Add functionality for ROC log tpr,fpr,auc data storage & plot comparison.
    Add functionality for multiple train/test cycles, where the SeqConf's are averaged together.

"""

#Model Choice
model = 'pscnn_v1'

#Data Manipulation
train_datasets = ['GaTechTr']
test_datasets = ['GaTechTe']
image_depth = 100
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
savefolderpath = 'SeqConfSaves/trial1'
verbose = 1

#Display
threshold = .99


if __name__ == "__main__":
    for train_dataset in train_datasets:
        for test_dataset in test_datasets:
            
            
            tk.createConfigFile(model, train_dataset, test_dataset, savefolderpath, threshold)

            
            #Returns dict with these Hyperparameters as keys.
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
            
            
            X_Data_train, Y_Data_train, X_Data_test, Y_Data_test, dncr, DataSpecs = tk.initDatasets(
                                                                                                    traindataset=train_dataset, 
                                                                                                    testdataset=test_dataset, 
                                                                                                    image_depth=image_depth, 
                                                                                                    negativewires=negativewires, 
                                                                                                    Hyperparameters=Hyperparameters
                                                                                                    )
            
                        
            
            
            
            
            
            
            train(X_Data_train, Y_Data_train, model, Hyperparameters)
            predictions = test(X_Data_test)
            
            
            tk.showPredictionsAuto(predictions)
            tk.savePredictionsAuto(predictions)
            
            
            tk.plotROCAuto(Y_Data_test, predictions, dncr)
            tk.plotROCAuto(Y_Data_test, predictions, dncr, save=True)
            

            
            
            #tk.showPredictions()
            #tk.savePredictions()
            #tk.plotROCLog()
            #tk.plotROC()
            #tk.plotConfusionMatrix()
            #tk.saveModel(trainedmodel)
            #... and so on
            
            
            
            
            
            
            
            
            

