# Tahoe: Tree Structure-Aware High Performance Inference Engine for Decision Tree Ensemble on GPU
Tahoe enables high-performance GPU inference for decision tree ensemble which can be used in high performance computing, advertising systems, medical diagnosis, financial fraud detection, etc.

## The easiest way to test it

## Run the SVHN data set as an example.
```
sh ./run_an_example.sh
```

## Run the 15 data set. (a little bit time-consuming)
```
sh ./run_all_15_examples.sh
```

# ---------------MORE DETAILS FOR TESTING-----------------
## Requirement (pass-test)
- Ubuntu 16.04
- CUDA 10.1

## Download trained forest and inference data
```
# Take the SVHN data set as an example.
# Download trained decision tree model
https://drive.google.com/file/d/1Z04IvVIIZdjUGZGIf4CpUQ2Ll06JVVll/view?usp=sharing
# Download data set
https://drive.google.com/file/d/1sqnuc8O9c58qphMHYIaNZSiHdDKV0AMF/view?usp=sharing

# In particular, all 15 trained models and data can be downloaded here (not necessary for testing)
https://drive.google.com/drive/folders/1S9ohva-P6NPB2GW8E1kkMqpQ-KmLL-E0?usp=sharing

# If you want to train the model from scratch (not necessary for testing)
Please refer to our another repo: https://github.com/zhen-xie/Decision-tree-ensemble.git
```

## Build/Install from Source
```
git clone https://github.com/zhen-xie/Tahoe.git
cd Tahoe;
make Tahoe;
```

## Run an example
```
# Run Tahoe with the corresponding paths. The first parameter is the model path, and the second parameter is the data path.
./Tahoe [PATH_MODEL] [PATH_DATASET]
```

## Output
```
[zxie10@gpu050 Tahoe]$ ./Tahoe /home/zxie10/Project1/model_SUSY.txt /home/zxie10/Project1/data_SUSY.txt;
Performance model choose #1 strategy.
/home/zxie10/Project1/model_SUSY.txt
Load forest
Load data
Predict on CPU and get standard results...
Test on GPU...
time_dense is 0.990169 us
results are correct
Using strategy 0
time_dense is 1.082617 us
results are correct
Using strategy 0
time_dense is 3.838786 us
results are correct
Using strategy 1
time_dense is 0.120020 us
results are correct
Using strategy 3
time_dense is 0.212903 us
results are correct
Enumerate al strategies and choose #1 strategy.
Performance model predicts correctly
Tahoe brings 8.25x speedup.
```

## Tahoe also provides a C++ interface to load forest and data from files and computes the prediction on GPU:
```C++
# Setup input forest, dataset, and algorithm, and construct the interface of Tahoe framework
BaseTahoeTest* pTest = new BaseTahoeTest(argv[1], argv[2]);
# Launch the inference of tree traversal
pTest->SetUp();
# Free resources
pTest->Free();
```
