# WIMEETSENSE
WIMEETSENSE: WiFi-Based Meeting Dataset for Human Activity Recognition during Online Meetings. The dataset is collected on a large scale with 33 participants in 5 different locations in 46 different experiment setups. 

This repository contains the reference code for data preprocessing and benchmarking. 
## How to download the dataset and code repository 
Clone the repository and enter the folder with the Python code:
```bash 
https://doi.org/10.5281/zenodo.11551205
```
Create "WIMEETSENSE" folder in your desired directory. 
```bash
mkdir WIMMETSENSE
cd WIMMETSENSE

```


The dataset contains CSI, video, and audio data with descriptions of filenames. This code repository can be used with CSI data. The CSI data consist of two zip files, "csi_semi.zip" and "csi_wild.zip". Unzip the files and place them at ```./WIMEETSENSE/code/raw_csi```. 

The code for WIMEETSENSE is implemented in Python, and all required directories are created inside the project folder. The project repository contains code for preprocessing the data and training and evaluating 9 state-of-the-art algorithms such as Constructive model, XGBoost (XGB),
Contrastive model, Support Vector Machine (SVM), Gradient Boosting (GB), Random Forest
(RF), Multiclass Logistic Regression (LR), attention-based Bi-directional Long Short-Term258
Memory(BLSTMA) and Sensing Human Activities through WiFi Radio Propagation (SHARP) for Human Activity Recognition (HAR).    

 The project folder consists of two subfolders, ```sharp``` for SHARP and ```code``` for other state-of-the-art algorithms. The HAR algorithms are evaluated in 4 setups: 
 
 $\mathbf{S1}$) in a simple setup, using the first 80% of each participant's data and the remaining 20% as testing, 
 
 $\mathbf{S2}$) 10-fold cross-validation setup, each time use the 9 fold as training and remaining $1$ fold for testing, 
 
 $\mathbf{S3}$) leave a few out, use a few participant or location data for testing, and the remaining for training, and 
 
 $\mathbf{S4}$) leave one setting out, use semi-controlled setting data for training and in-the-wild setting for testing, alternatively use the in-the-wild setting for training and semi-controlled setting data for testing.
 ## Create sub directory, required to run the code 
 ### Goto ```code``` directory 
 ```bash
 mkdir model -p preprocessing/preprocessed_amp/semi_amp preprocessing/preprocessed_amp/wild_amp model raw/semi raw/wild training_testing_data/S1 training_testing_data/S3 training_testing_data/S4
 ```
 ### Goto ```sharp_code``` directory 
 ```bash
 mkdir model -p S1 S2 S3 preprocessing/semi preprocessing/wild processed/semi processed/wild preprocess/semi preprocessed/wild model input_combine/semi input_combine/wild dopplers/semi dopplers/wild
 ```

 
 ## The following scripts preprocess the raw data, training and evaluating HAR algorithms. 
 The following scripts preprocess, train, and evaluate the  Constructive model, XGBoost (XGB),
Contrastive model, Support Vector Machine (SVM), Gradient Boosting (GB), Random Forest
(RF), Multiclass Logistic Regression (LR) and attention-based Bi-directional Long Short-Term258
Memory(BLSTMA). 
### Preprocessing
Note: Navigate to ```code``` directory. 

Computing amplitude feature and denoising 
```bash
python amplitude_preprocessing.py
```
### Creating training and testing setup
Create training and testing dataset for ```S1```, ```S2``` and ```S3```. The training scripts contain code for ```S2``. 
```bash
python training_testing.py
```
### Training and evaluation
```bash
python logistic.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python logistic.py S1
```bash
python svm.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python svm.py S1
```bash
python gradient_boosting.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python gradient_boosting.py S1
```bash
python random_forest.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python random_forest.py S1
```bash
python xgboost.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python xgboost.py S1
```bash
python bilstm.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python bilstm.py S1
```bash
python contrastive.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python contrastive.py S1
```bash
python constructive.py <'write evaluation setup such as S1, S2, S3'>
```
e.g., python constructive.py S1

##  The following scripts preprocess the raw data, train and evaluate the SHARP.
Note: Navigate to ```sharp_code``` directory. 
### Preprocessing 
Convert the raw CSI values into complex number. 
```bash
python convert_complex.py
```
Combine CSI files participants-wise and store them into ```input_combine```

```bash
python combine.py
```
### Denoising- Phase saniization

```bash
python preprocessing.py
python CSI_signal_Hestimation.py
python CSI_signal_reconstruction.py
```
Computing Doppler vector feature features and denoising 
```bash
python CSI_doppler_computation.py
```
### Creating training and testing setup
```bash
python training_testing.py
```
```bash
python CSI_doppler_create_dataset_train_test.py ./ <'setup> <Number of packets in a sample> <Number of packet for sliding operations> <Number of samples per window> <Number of samples to stride> <labels_activities> <Number of streams * number of antennas>
```
e.g., python CSI_doppler_create_dataset_train_test.py ./ S4 31 1 256 30 Fo,LD,LL,LR,LU,No,Sh 1


Fo - Looking Forward, LD - Looking Down, LL - Looking Left, LR - Looking Right, LU - Looking Up No - Nodding, Sh - Shaking
### Training and Evaluating the SHARP model 
```bash
python CSI_network.py ./ <setup> <feature_length-set by doppler_computation.py code> <Length along the time dimension (width)> <Number of channel> <Number of samples in a batch> <Number of antenna * number of spatial streams> <Name base> <labels_activities>
```
e.g., python CSI_network.py ./ S4 100 340 1 16  1 temp  Fo,LD,LL,LR,LU,No,Sh
## Python and relevant libraries version
Python >= 3.7.7

TensorFlow >= 2.6.0

Numpy >= 1.19.5

Scipy = 1.4.1

Scikit-learn = 0.23.2

OSQP >= 0.6.1





