# Landmine Detection Model

This project is designed to train and test various deep learning models for landmine detection using different datasets.

## Features
- Supports multiple models: `resnet18`, `resnet38`, `pscnn_v1`, `resnetinspired`
- Uses training and testing datasets for evaluation
- Configurable hyperparameters for model training
- Automated data processing, training, testing, and visualization
- Supports ROC curve generation and other performance metrics

## Installation
Ensure you have Python installed and the necessary dependencies:
```bash
pip install -r requirements.txt
Usage
Run the script to start training and testing:
python main.py


Configuration
Modify the config.py file or update parameters in main.py:

train_datasets: Specify training datasets
test_datasets: Specify testing datasets
model: Choose a model
activationFunction, loss, num_filters, epoch_num, optimizer: Model parameters
threshold: Set the classification threshold


Known Issues
plotOverlay() is not working correctly.



Improvements needed for:
Storing and comparing ROC log data
Averaging multiple train/test cycles
Results & Visualization
The script generates:



Predictions stored automatically
ROC Curves for model performance evaluation
Confusion Matrices for error analysis
