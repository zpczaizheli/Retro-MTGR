# Retro-MTGR

## Title
Retro-MTGR: Molecule Retrosynthesis Prediction via Multi-Task Graph Representation Learning

## Overview

 ![image](https://github.com/zpczaizheli/Retro-MTGR/assets/47655168/ccc99491-49fc-4c5b-ad34-94a3c75bffb3)

## Environment Requirements
- Python = 3.7.8
- numpy = 1.21.6
- pytorch = 1.13.1
- rdkit = 2022.03.1
- Matplotlib = 3.5.1

## Process
### Step 1: Data Processing
The goal of this step is to convert the raw data into a format that Retro-MTGR can recognize.  
- Navigate to the `original Data` folder:
```bash
cd Retro-MTGR
```
- If the target is the uspto-50k dataset, run the following code:
```bash
python 50k-Data processing.py
```
- If the target is the uspto-MIT dataset, run the following code:
```bash
python python mit-Data processing.py
```
Note: You need to modify the input file names in the code to select the files you need to process, such as `test-MIT.txt`, `train-MIT.txt`, and `valid-MIT.txt`.
After the data processing is complete, the processed data needs to be moved to the data directory in the parent directory for the next step.

### Step 2: Model Training and Testing
- if you want to train and test the Retro-MTGR model, you need to run the following code:
```bash
cd Retro-MTGR
python Train_Model-5.0.py
```
Note: You need to modify the target data file name in the code to select the data to be executed.
- For example, if your target data is all the uspto-50k data, you need to change the file name input in the 'Train_Model-5.0.py' file to 'uspto50k-alldata.txt':
```bash
Datapath = 'data/USPT-50K/class8-.txt'
```
- Additionally, you need to specify the range of the training and testing sets by using arrays:

```bash
Train_list = list(range(0, 300))
Test_list = list(range(300, 400))
```
## Output
Once the model training is completed, you will get two files:

- Test-predict.txt: Contains the retrosynthesis prediction results for the target molecules in the test set.

- Test-result.txt: Contains detailed performance data of the model.



