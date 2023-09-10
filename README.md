# MNIST_classifier
The implementation of the MNIST classifier model using only numpy

You can download the MNIST dataset with the download_mnist.py script:
```bash
cd Preprocessing
python download_mnist.py
cd ..
```
Default download folder is set to ../mnist but can be changed in a script
&nbsp;
&nbsp;

First run the script setup.py to compile Cython convolution functions:
```bash
python setup.py build_ext --inplace
```
&nbsp;
&nbsp;
You can train the model with the train.py script:
```bash
python train.py -data_path mnist -lr 0.1 -batch_size 100 -num_epochs 8 -random_state 42
```
&nbsp;
&nbsp;
You can test the model performances with the test.py script, pretrained weights are also provided in the ModelWeights folder:
```bash
python test.py -data_path mnist -data_path mnist -weights_path ModelWeights/MNIST_classifier_convolution_8_best_val_weights.pkl -random_state 42
```





