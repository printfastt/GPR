import numpy as np
import matplotlib.pyplot as plt
import os
import config
import seaborn as sns
import importlib
import subprocess
from sklearn.metrics import roc_curve, auc


    
def createConfigFile(model, train_dataset, test_dataset, savefolderpath, threshold, filepath="config.py"):
    config_content = f"""# Configuration File

model = '{model}'
train_dataset = '{train_dataset}'
test_dataset = '{test_dataset}'
savefolderpath = 'SeqConfSaves/{savefolderpath}'
threshold = {threshold}
"""
    
    with open(filepath, "w") as file:
        file.write(config_content)
    print(f"Config file created at {filepath}")

    if os.path.exists(filepath):
        importlib.reload(config)






def getPath():
    if hasattr(config, 'savefolderpath') and hasattr(config, 'train_dataset') and hasattr(config, 'test_dataset'):
        return f"{config.savefolderpath}/TR_{config.train_dataset}TE_{config.test_dataset}"
    else:
        raise AttributeError("Config attributes are missing: Ensure savefolderpath, train_dataset, and test_dataset are set.")







#Creates a dict and returns it with the given arguments as keys.
def initHyperparameters(model, activationFunction, num_filters, epoch_num, kSize, dropoutAmount, batch_size, optimizer, loss, iterations, verbose):
        Hyperparameters = {
            "num_filters": num_filters,
            "epoch_num": epoch_num,
            "kernelSize": (kSize,1,kSize),
            "activationFunction": activationFunction,
            "dropoutAmount": dropoutAmount,
            "batch_size" : batch_size,
            "optimizer" : optimizer,
            "loss" : loss,
            "iterations" : iterations,
            "verbose" : verbose,
            "model" : model
        }
        return Hyperparameters
    
    
    
    
  
    
  
    
  
    
"""
initDatasets is responsible for taking in names of datasets, initializing, processing, labeling, and then returning the datasets.

Options:
    negativewires: if false, wires in 1022,0914,0524, and 1014 are included in the dataset as a non-target.
                   if true, wires are simply excluded from the dataset.
                   NOTE: needs to be setup.
                   NOTE: rename to 'RemoveWires'
    
    trainrange:    if trainrange is passed, it will use that range, otherwise it will use default landmines only.
    
    image_depth:   if image_depth is passed, it will cut the image depth off at the given int.
    
    
    
    NOTE: can simplify code by doing the randomization portion of in case 914cp1022cp1014cp0524cp and 914hh1022hh1014hh0524hh as one.
    NOTE: need to make subfunctions for initDatasets; too large right now.
    NOTE: maybe add test/traindatasetFN (full name)
    NOTE: currently you cannot adjust the image depth on GaTechTr.
"""
def initDatasets(**kwargs):
    grouping = 2970
    
    if 'negativewires' in kwargs:
        negativewires = kwargs['negativewires']
        
    if 'trainrange' in kwargs:
        trainrange = kwargs['trainrange']
        start = trainrange[0]
        end = trainrange[1]
        
        
        
    if 'traindataset' in kwargs:
        traindataset = kwargs['traindataset'].lower()
        
#Defines which datasets are the original 4
        DatasetsOriginal = {
            '1022cp': (7,14),
            '0914cp': (7,14),
            '0524cp': (4,10),
            '1014cp': (4,11),
            '1022hh': (7,14),
            '0914hh': (7,14),
            '0524hh': (4,10),
            '1014hh': (4,11)
            }
#Defines which datasets are combined
        DatasetsCombined = {
            '914cp1022cp1014cp0524cp' : ((7,13),(18,25),(30,41)),
            '914hh1022hh1014hh0524hh' : ((7,13),(18,25),(30,41))

            }
     
#Train dataset selection        
        if kwargs['traindataset'] in DatasetsOriginal:
            match(traindataset):
                    case '1022cp':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20201022cpDp200SclExt.npy')
                        start, end = DatasetsOriginal['1022cp']
                        traindatasetNN = '1022cp'
                    case '0914cp':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20200914cpDp200SclExt.npy')
                        start, end = DatasetsOriginal['0914cp']
                        traindatasetNN = '0914cp'
                    case '1022hh':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20201022hhDp200SclExt.npy')
                        start, end = DatasetsOriginal['1022hh']
                        traindatasetNN = '1022hh'
                    case '0914hh':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20200914hhDp200SclExt.npy')
                        start, end = DatasetsOriginal['0914hh']
                        traindatasetNN = '0914hh'
                    case '0524cp':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20220524cpDp200SclExt.npy')
                        start, end = DatasetsOriginal['0524cp']
                        traindatasetNN = '0524cp'
                    case '1014cp':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20211014cpDp200SclExt.npy')
                        start, end = DatasetsOriginal['1014cp']
                        traindatasetNN = '1014cp'
                    case '0524hh':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20220524hhDp200SclExt.npy')
                        start, end = DatasetsOriginal['0524hh']
                        traindatasetNN = '0524hh'
                    case '1014hh':
                        X_Data_train = np.load('LandMine_Data/Train/2dDataTrain20211014hhDp200SclExt.npy')
                        start, end = DatasetsOriginal['1014hh']
                        traindatasetNN = '1014hh'
                    case _:
                        print("Data init issue")
                        
            num_obj = X_Data_train.shape[0]
            Y_Data_train = np.zeros((num_obj), dtype = np.uint8)
            for x in range(grouping * start, grouping * end):
                Y_Data_train[x] = 1
                
            true_indices = np.where(Y_Data_train == 1)[0]
            num_true = len(true_indices)
            
            if negativewires == False:
                false_indices = np.where(Y_Data_train == 0)[0]
            else:
                Y_Data_train_no_wires = Y_Data_train[grouping*start:]
                false_indices = np.where(Y_Data_train_no_wires == 0)[0]
        
            selected_false_indices = np.random.choice(false_indices, size = num_true, replace = False)
            X_Data_train = np.concatenate([X_Data_train[true_indices], X_Data_train[selected_false_indices]])
            
            Y_Data_train = np.zeros((2 * num_true,), dtype=np.uint8)
            Y_Data_train[:num_true] = 1
                
        elif(kwargs['traindataset'] in DatasetsCombined):
        #Handles datasets other then 1022 ,0914, 0524 or 1014.
            print("Not a 1022, 0914, 0524, or 1014 dataset.")
            match(traindataset):
                case '914cp1022cp1014cp0524cp':
                    X_Data_train = np.load('LandMine_Data/Train/914cp_1022cp_1014cp_524cp_combined.npy')
                    num_obj = X_Data_train.shape[0]
                    Y_Data_train = np.zeros((num_obj), dtype=np.uint8)
                    for x in range(grouping * 7, grouping * 13):
                        Y_Data_train[x] = 1
                    for x in range(41580 + grouping * 7, 41580 + grouping * 13):
                        Y_Data_train[x] = 1
                    for x in range(83160 + grouping * 5, 83160 + grouping * 12):
                        Y_Data_train[x] = 1
                    for x in range(115830 + grouping * 5, 115830 + grouping * 11):
                        Y_Data_train[x] = 1
                    traindatasetNN = '2dDataTrain0914cp1022cp1014cp0524cp'
                case '914hh1022hh1014hh0524hh':
                    X_Data_train = np.load('LandMine_Data/Train/914hh_1022hh_1014hh_524hh_combined.npy')
                    num_obj = X_Data_train.shape[0]
                    Y_Data_train = np.zeros((num_obj), dtype=np.uint8)
                    for x in range(grouping * 7, grouping * 13):
                        Y_Data_train[x] = 1
                    for x in range(41580 + grouping * 7, 41580 + grouping * 13):
                        Y_Data_train[x] = 1
                    for x in range(83160 + grouping * 5, 83160 + grouping * 12):
                        Y_Data_train[x] = 1
                    for x in range(115830 + grouping * 5, 115830 + grouping * 11):
                        Y_Data_train[x] = 1
                    traindatasetNN = '2dDataTrain0914hh1022hh1014hh0524hhDp200SclExt'
                case _:
                    print("Data init issue")
                    
            true_indices = np.where(Y_Data_train ==0)[0]
            num_true = len(true_indices)
            
            if negativewires == False:
                false_indices = np.where(Y_Data_train == 0)[0]
            else:
                Y_Data_train_no_wires = Y_Data_train[grouping*start:]
                false_indices = np.where(Y_Data_train_no_wires == 0)[0]
        
            selected_false_indices = np.random.choice(false_indices, size = num_true, replace = False)
            X_Data_train = np.concatenate(X_Data_train[true_indices], X_Data_train[selected_false_indices])
            
            Y_Data_train = np.zeros((2 * num_true,), dtype=np.uint8)
            Y_Data_train[:num_true] = 1
                
                
        else:
            match(traindataset):
                case 'gatechtr':
                    X_Data_train = np.load('LandMine_Data/Train/2dDataTrainGaTech.npy')
                    num_obj = X_Data_train.shape[0]
                    Y_Data_train = np.zeros((num_obj),dtype=np.uint8)
                    for x in range(int(num_obj/2)):
                        Y_Data_train[x] = 1
                    traindatasetNN = 'GaTechTr'
                case _:
                    print("Data init issue")
                    
                    
                    
                    
#Test dataset selection                    
        if 'testdataset' in kwargs:
            testdataset = kwargs['testdataset'].lower()
            match (testdataset):
                case '1022cp':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20201022cpSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20201022_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20201022_dncr.npy')
                    testdatasetNN = '20201022cp'
                case '0914cp':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20200914cpSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20200914_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20200914_dncr.npy')
                    testdatasetNN = '20200914cp'
                case '0524cp':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20220524cpSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20220524_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20220524_dncr.npy')
                    testdatasetNN = '20220524cp'
                case '1014cp':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20211014cpSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20211014_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20211014_dncr.npy')
                    testdatasetNN = '20211014cp'
                case '1022hh':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20201022hhSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20201022_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20201022_dncr.npy')
                    testdatasetNN = '20201022hh'
                case '0914hh':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20200914hhSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20200914_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20200914_dncr.npy')
                    testdatasetNN = '20200914hh'
                case '0524hh':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20220524hhSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20220524_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20220524_dncr.npy')
                    testdatasetNN = '20220524hh'
                case '1014hh':
                    X_Data_test = np.load("LandMine_Data/Test/3dDataSeq20211014hhSemiThp05Dp200.npy")
                    Y_Data_test = np.load('LandMine_Data/Truth/GT20211014_mask.npy')
                    dncr = np.load('LandMine_Data/Truth/GT20211014_dncr.npy')
                    testdatasetNN = '20211014hh'
                case 'gatechte':
                    X_Data_test = np.load("LandMine_Data/Test/2dDataSeqGaTech.npy")
                    Y_Data_test = np.zeros(X_Data_test.shape[0])
                    dncr = np.zeros(X_Data_test.shape[0])
                    testdatasetNN = 'GaTechTe'
                case _:
                    print("data init issue")
        
#Sets image depth                    
    if 'image_depth' in kwargs:
        image_depth = kwargs['image_depth']
    
        if isinstance(image_depth, tuple):  
            min_depth, max_depth = image_depth
        else:
            min_depth, max_depth = 0, image_depth  
    
        if testdataset in DatasetsOriginal or testdataset in DatasetsCombined:
            X_Data_train = setImageDepth(X_Data_train, min_depth, max_depth)
            X_Data_test = setImageDepth(X_Data_test, min_depth, max_depth)
        else:
            print("*****\n*****\nWARNING: SET IMAGE DEPTH ON NON-ORIGINAL 4 TEST DATASET\n*****\n*****")
            X_Data_test = setImageDepth(X_Data_test, min_depth, max_depth)
            X_Data_train = setImageDepth(X_Data_train, min_depth, max_depth)
            
                                
            
    if traindataset in DatasetsOriginal:
        trainrange = DatasetsOriginal[traindataset]
    elif traindataset in DatasetsCombined:
        trainrange = DatasetsCombined[traindataset]
    else:
        trainrange = '0, num_obj/2'
        
    DataSpecs = {
        'TrainDataSet': traindatasetNN,
        'TestDataSet': testdatasetNN,
        'TeDS': testdataset,
        'TrDS': traindataset,
        'TrainRange': trainrange,
        'NegativeWires': negativewires,
        'image_depth': image_depth,
    }

    
    
    specsListAuto(DataSpecs, kwargs['Hyperparameters'])
    
    
    
    return X_Data_train, Y_Data_train, X_Data_test, Y_Data_test, dncr, DataSpecs

    

def setImageDepth(dataset, min_depth, max_depth):
    dataset = dataset[:, min_depth:max_depth, :, :]
    return dataset

def removeDimension(dataset):
    dataset = dataset.reshape(dataset.shape[0],dataset.shape[1],dataset.shape[3])
    return dataset




"""
Options:
    offset: starts displaying graph at the given index.
"""
def visualizeDataset(dataset, **kwargs):
    if dataset.ndim == 4:
        dataset = removeDimension(dataset)
        
    offset = kwargs.get('offset', 0) 
    for i in range(offset, dataset.shape[0]): 
        plt.imshow(dataset[i], cmap='gray', aspect='auto')
        plt.title(f"Image {i+1}/{dataset.shape[0]}")
        plt.colorbar()
        plt.show()
        



def specsListAuto(DataSpecs, Hyperparameters):
    base_save_dir = config.savefolderpath
    folder_name = f"TR_{config.train_dataset}TE_{config.test_dataset}"
    folder_path = os.path.join(base_save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    image_filename = f"TR_{config.train_dataset}TE_{config.test_dataset}.png"
    image_save_path = os.path.join(folder_path, image_filename)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Dataset and Hyperparameters Overview")
    ax.axis('off')
    specs_text = "\n".join([
        f"Train Dataset: {config.train_dataset}",
        f"Test Dataset: {config.test_dataset}",
        f"Train Range: {DataSpecs['TrainRange']}",
        f"Image Depth: {DataSpecs['image_depth']}",
        f"Negative Wires: {DataSpecs['NegativeWires']}",
        "",
        "Hyperparameters:",
        f"Number of Filters: {Hyperparameters['num_filters']}",
        f"Epochs: {Hyperparameters['epoch_num']}",
        f"Kernel Size: {Hyperparameters['kernelSize']}",
        f"Activation Function: {Hyperparameters['activationFunction']}",
        f"Dropout: {Hyperparameters['dropoutAmount']}",
        f"Batch Size: {Hyperparameters['batch_size']}",
        f"Optimizer: {Hyperparameters['optimizer']}",
        f"Loss Function: {Hyperparameters['loss']}",
        f"Iterations: {Hyperparameters['iterations']}",
    ])
    ax.text(0.5, 0.5, specs_text, fontsize=12, va='center', ha='center', wrap=True)
    plt.savefig(image_save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
   
    
   
def savePredictions(predictions):
    np.save(getPath() + f'/TR_{config.train_dataset}TE_{config.test_dataset}SeqCnf.npy', predictions)



""" plotPredictions()
Generates and optionally saves/displays predictions as images.

    Options:
        save (bool)       : If True, saves the images instead of just displaying them. Defaults to False.
        show (bool)       : If True, displays the images. Defaults to True.
        threshold (float) : If provided, applies a threshold filter to the predictions. Defaults to config.threshold.
        reshape (tuple)   : If provided, reshapes the predictions before processing. Defaults to (250, 320) if not in 'gatechte' dataset.
        model (str)       : Custom model name for the plot title. Defaults to config.model.
        file (str)        : If provided, loads predictions from the specified file path instead of using the function argument.
        test_dataset (str): Specifies which test dataset is being used. Defaults to config.test_dataset.
        train_dataset (str): Specifies which train dataset is being used. Defaults to config.train_dataset.
        
        
    Notes:
        - If `file` is provided, predictions are loaded from the specified file.
        - If `reshape` is provided, the predictions are reshaped accordingly.
        - The function flips the predictions vertically before displaying.
        - The image is saved with filenames based on train and test dataset names.
        - A thresholded version of the prediction is also saved if `save=True`.
"""
def plotPredictions(predictions, **kwargs):
    if 'file' in kwargs:
        seqcnf = np.load(kwargs['file'])
    elif isinstance(predictions, str):
        seqcnf = np.load(predictions)
    else:
        seqcnf = predictions

    if 'reshape' in kwargs:
        seqcnf = seqcnf.reshape(kwargs['reshape'][0], kwargs['reshape'][1])
    else:
        test_dataset = kwargs.get('test_dataset', config.test_dataset).lower()
        if test_dataset == 'gatechte':
            seqcnf = seqcnf.reshape(13832).reshape(104, 133)
        else:
            seqcnf = seqcnf.reshape(80000).reshape(250, 320)

    seqcnf_mirrored = np.flipud(seqcnf)

    save_folder = getPath()
    os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists

    train_dataset = kwargs.get('train_dataset', config.train_dataset)
    test_dataset = kwargs.get('test_dataset', config.test_dataset)
    base_filename = f"TR_{train_dataset}TE_{test_dataset}"
    original_filename = os.path.join(save_folder, f"{base_filename}Original.png")
    thresholded_filename = os.path.join(save_folder, f"{base_filename}Thresholded.png")

    plt.figure()
    model = kwargs.get('model', config.model)
    plt.title(f'SeqConf({model})')
    plt.grid(False)
    plt.imshow(seqcnf_mirrored, cmap='viridis')
    plt.colorbar()

    if kwargs.get('save', False):
        plt.savefig(original_filename, bbox_inches='tight')

    if kwargs.get('show', True):
        plt.show()
    else:
        plt.close()

    threshold = kwargs.get('threshold', float(config.threshold))
    seqcnf_filtered = np.where(seqcnf > threshold, seqcnf, 0)
    seqcnf_filtered_mirrored = np.flipud(seqcnf_filtered)

    plt.figure()
    plt.title(f'Filtered SeqConf ({model}) (Values > {threshold})')
    plt.grid(False)
    plt.imshow(seqcnf_filtered_mirrored, cmap='viridis')
    plt.colorbar()

    if kwargs.get('save', False):
        plt.savefig(thresholded_filename, bbox_inches='tight')

    if kwargs.get('show', True):
        plt.show()
    else:
        plt.close()


def plotPredictionsAuto(predictions):
    plotPredictions(predictions, save=True, show=True)
            
    
    
""" plotROC()
   Generates and optionally saves/displays an ROC curve.

   Options:
       save (bool)       : If True, saves the plot instead of just displaying it. Defaults to False.
       show (bool)       : If True, displays the plot. Defaults to True.
       dncr (array)      : Optional 'don't care' region, which removes irrelevant data points from ROC calculation.
       
   Notes:
   - Computes and plots an ROC curve with and without DNCR (don't care regions).
   - AUC values are calculated and displayed in the legend.
   - If the test dataset is 'GaTechTe', a warning is issued and the function exits.
"""
def plotROC(Y_Data_test, predictions, dncr = None, **kwargs):

    
    if config.test_dataset.lower() == 'gatechte':
        print("*****\n*****\nWARNING: NO Y_Data_test or dncr for GaTechTe \n*****\n*****")

    
    
    Y_Data_test = Y_Data_test.flatten()
    predictions = predictions.flatten()
    dncr = dncr.flatten()

    fpr, tpr, _ = roc_curve(Y_Data_test, predictions)
    roc_auc = auc(fpr, tpr)

    indexes = np.where(dncr == 1)
    Y_Data_test = np.delete(Y_Data_test, indexes)
    predictions = np.delete(predictions, indexes)

    fpr_dncr, tpr_dncr, _ = roc_curve(Y_Data_test, predictions)
    roc_auc_dncr = auc(fpr_dncr, tpr_dncr)

    plt.figure()
    plt.plot(fpr_dncr, tpr_dncr, color='darkorange', lw=2, 
             label=f'ROC curve with DNCR removed (AUC = {roc_auc_dncr:.6f})')
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.6f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with and without Don’t Care Region')
    plt.legend(loc="lower right")

    if kwargs.get('save', False):
        save_folder = getPath()
        os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(save_folder, f"TR_{config.train_dataset}TE_{config.test_dataset}ROC.png")
        plt.savefig(filename, bbox_inches='tight')
        

    if kwargs.get('show', True):
        plt.show()
    else:
        plt.close()
        
        
        
def plotROCAuto(Y_Data_test, predictions, dncr):
    plotROC(Y_Data_test, predictions, dncr, save=True, show=True)
    
    
    
    
    
    
""" plotROCLog()
    Generates and optionally saves/displays an ROC curve using a logarithmic scale for the false positive rate.

    Options:
        save (bool)       : If True, saves the plot instead of just displaying it. Defaults to False.
        show (bool)       : If True, displays the plot. Defaults to True.
        dncr (array)      : Optional 'don't care' region, which removes irrelevant data points from ROC calculation.

    Notes:
        - Computes and plots an ROC curve with a logarithmic scale on the x-axis.
        - AUC values are calculated and displayed in the legend.
        - A random classifier reference line is included.
        - If the test dataset is 'GaTechTe', a warning is issued and the function exits.
"""
def plotROCLog(Y_Data_test, predictions, dncr = None, **kwargs): 
    if config.test_dataset.lower() == 'gatechte':
        print("*****\n*****\nWARNING: NO Y_Data_test or dncr for GaTechTe \n*****\n*****")
        return
    
    
    
    Y_Data_test = Y_Data_test.flatten()
    predictions = predictions.flatten()
    dncr = dncr.flatten()

    fpr, tpr, _ = roc_curve(Y_Data_test, predictions)
    roc_auc = auc(fpr, tpr)

    indexes = np.where(dncr == 1)
    Y_Data_test = np.delete(Y_Data_test, indexes)
    predictions = np.delete(predictions, indexes)

    fpr_dncr, tpr_dncr, _ = roc_curve(Y_Data_test, predictions)
    roc_auc_dncr = auc(fpr_dncr, tpr_dncr)

    plt.figure()
    plt.plot(fpr_dncr, tpr_dncr, color='darkorange', lw=2, 
             label=f'ROC with DNCR removed (AUC = {roc_auc_dncr:.6f})')
    plt.plot(fpr, tpr, color='blue', lw=2, 
             label=f'ROC (AUC = {roc_auc:.6f})')
    plt.axline((1e-10, 0), slope=0, color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.00000001, 1.0])
    plt.xscale('log')
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title('Logarithmic ROC Curve with and without Don’t Care Region')
    plt.legend(loc="upper right", fontsize='small')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    if kwargs.get('save', False):
        save_folder = getPath()
        os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(save_folder, f"TR_{config.train_dataset}TE_{config.test_dataset}ROCLog.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if kwargs.get('show', True):
        plt.show()
    else:
        plt.close()
        
        
        
def plotROCLogAuto(Y_Data_test, predictions, dncr, **kwargs):
    plotROCLog(Y_Data_test, predictions, dncr, save = True, show = True)
    
    
    
def plotConfusionMatrix(Y_Data_test, predictions, dncr = None, **kwargs):
    
    """
    Generates and optionally saves/displays a confusion matrix.

    Options:
        save (bool)       : If True, saves the plot instead of just displaying it. Defaults to False.
        show (bool)       : If True, displays the plot. Defaults to True.
        dncr (array)      : Optional 'don't care' region, which modifies the confusion matrix calculations.
        discard (bool)    : If True, discards 'don't care' regions instead of incorporating them.
    """
    if config.test_dataset.lower() == 'gatechte':
        print("*****\n*****\nWARNING: NO Y_Data_test or dncr for GaTechTe \n*****\n*****")
        return
    
    threshold = config.threshold

    
    if "dncr" in kwargs:
        print("evaluating CM with DNCR region")
        predictions = np.array(predictions).flatten()
        Y_Data_test = np.array(Y_Data_test).flatten()
        dncr = np.array(dncr).flatten()
        dncr = np.where(dncr == 1, 2, 0)
        Y_Data_test = Y_Data_test + dncr
        predictions = np.where(predictions > threshold , 1, 0) 
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        
        if "discard" in kwargs:
            for pred, true in zip(predictions, Y_Data_test):
                if pred == 1 and true == 2 or pred == 0 and true == 2:
                    continue
                elif pred == 0 and true == 0:
                    true_negative += 1
                elif pred == 1 and true == 1:
                    true_positive += 1
                elif pred == 1 and true == 0:
                    false_positive += 1
                elif pred == 0 and true == 1:
                    false_negative += 1
            type = "discard"
        else:
            for pred, true in zip(predictions, Y_Data_test):
                if pred == 0 and true == 0:
                    true_negative += 1
                elif pred == 1 and true == 1:
                    true_positive += 1
                elif pred == 1 and true == 0:
                    false_positive += 1
                elif pred == 0 and true == 1:
                    false_negative += 1
                elif pred == 1 and true == 2:
                    true_positive +=1
                elif pred == 0 and true == 2:
                    true_negative += 1
            type = "ignore FN,FP"

        confusion_matrix = np.array([[true_positive, false_positive],
                                     [false_negative, true_negative]])

    else:
        predictions = np.array(predictions).flatten()
        Y_Data_test = np.array(Y_Data_test).flatten()
        predictions = np.where(predictions > config.threshold, 1, 0) 
        
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
    
        for pred, true in zip(predictions, Y_Data_test):
            if pred == 0 and true == 0:
                true_negative += 1
            elif pred == 1 and true == 1:
                true_positive += 1
            elif pred == 1 and true == 0:
                false_positive += 1
            elif pred == 0 and true == 1:
                false_negative += 1
        
        confusion_matrix = np.array([[true_positive, false_positive],
                                     [false_negative, true_negative]])
        type = f"at {threshold}"
    
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                yticklabels=['True', 'False'],
                xticklabels=['True', 'False'])
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Confusion Matrix {type}')
    
    if kwargs.get('save', False):
        save_folder = getPath()
        os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(save_folder, f"TR_{config.train_dataset}TE_{config.test_dataset}ConfMatrix.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if kwargs.get('show', True):
        plt.show()
    else:
        plt.close()

def plotConfusionMatrixAuto(Y_Data_test, predictions, dncr, **kwargs):
    plotConfusionMatrix(Y_Data_test, predictions, dncr, save=True, show=True, **kwargs)



"""
    Generates and optionally saves/displays an overlay of predictions and ground truth.

    Options:
        save (bool)       : If True, saves the plot instead of just displaying it. Defaults to False.
        show (bool)       : If True, displays the plot. Defaults to True.
        dncr (array)      : Optional 'don't care' region, which modifies the overlay calculation.

    Notes:
        - The overlay visualizes the sum of `Y_Data_test` and `predictions`.
        - If `dncr` is provided, it is incorporated into `Y_Data_test` before overlaying.
        - The plot is flipped vertically before displaying.
        - The image is saved with filenames based on train and test dataset names.
"""
def plotOverlay(Y_Data_test, predictions, dncr=None, **kwargs):
    
    Y_Data_test = Y_Data_test.reshape(250, 320)
    predictions = predictions.reshape(250, 320)
    
    combined = Y_Data_test + predictions
    combined_mirrored = np.flipud(combined)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(combined_mirrored)
    axes[0].set_title('Overlay: predictions + Y_Data_test')
    
    if dncr is not None:
        dncr = dncr.reshape(250, 320)
        dncr = np.where(dncr == 1, 2, 0)
        Y_Data_test = Y_Data_test + dncr
    
    combined = Y_Data_test + predictions
    combined_mirrored = np.flipud(combined)
    
    axes[1].imshow(combined_mirrored)
    axes[1].set_title('Modified Overlay (with dncr)')
    
    if kwargs.get('save', False):
        save_folder = getPath()
        os.makedirs(save_folder, exist_ok=True)  # Ensure the directory exists
        filename = os.path.join(save_folder, f"TR_{config.train_dataset}TE_{config.test_dataset}Overlay.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if kwargs.get('show', True):
        plt.show()
    else:
        plt.close()

def plotOverlayAuto(Y_Data_test, predictions, dncr=None, **kwargs):
    plotOverlay(Y_Data_test, predictions, dncr, save=True, show=True, **kwargs)
    
    
    

def createPPTX(autocreate, pptxEnvPath):
    if autocreate == True: 
        pptx_script_path = "makePresAuto.py"   
        subprocess.run([pptxEnvPath, pptx_script_path])
    else:
        return

    

    
    
        
        
    
    
    
            
            
        
    
                
                
                

                
            
        
        
    