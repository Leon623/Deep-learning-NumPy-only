# MNIST_classifier
The implementation of the MNIST classifier model using only numpy

This project was built using Python 3.10.12 version. To install the requirements use:
```bash
pip install -r requirements.txt
```

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
In the Results folder there are training results for different hyperparameters stored as:
**[model.name]\_[num_epochs]\_[batch_size]\_[lr]\_[droupout_rate]\_[random_state].png**
&nbsp;
&nbsp;
<br/>
<br/>
You can test the model performances with the test.py script, pretrained weights are also provided in the ModelWeights folder stored as : **[model.name]\_[num_epochs]\_[batch_size]\_[lr]\_[droupout_rate]\_[random_state]_best_val_weights.pkl**.
```bash
python test.py -data_path mnist -data_path mnist -weights_path ModelWeights/MNIST_classifier_convolution_8_100_0.1_0.0_42.pkl -random_state 42
```
&nbsp;
&nbsp;
You can use grid_search for hyperparameters optimization with the grid_seach.py script:
```bash
python grid_search.py -num_epoch 8,10 -batch_size 100,200,280 -dropout_rate 0.0,0.25,0.5 -learning_rate 0.1,0.05 ModelWeights/MNIST_classifier_convolution_8_100_0.1_0.0_42_best_val_weights.pkl -random_state 42
```
&nbsp;
&nbsp;
<br/>
Example for one training, testing and visualization is provided in the Train_test.ipynb jupyter notebook.
<br/>
Documentation for every script is available in the Documentation folder










