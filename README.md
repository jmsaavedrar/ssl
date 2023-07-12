# SSL:  A repository for self-supervised learning

This repository contains implementations of SimSiam and BYOL, which were tested with the quickdraw dataset. 
To make the data loading easy, we use [TFDS](https://www.tensorflow.org/datasets/add_dataset). TFDS allows us to load popular datasets as ImageNet or build our own datasets. In the next lines, we provide simple steps to run this code. 


## Dataset
We have prepared a subset of QuickDraw with 1100 instances per class. You can download it from [here](https://www.dropbox.com/s/eq3vzu65elii62i/tfds_qd.tar).
You should unpack it into  $HOME/tensorflow_datasets (by default, python will look the datasets there). However, you can put the data into another folder, but you'll need to specify the data_dir in the tfds.load function in the [train_ssl.py](train_ssl.py) file.

In addition, we add configuration files that may facilitate experimentation. In our example we use [qd.ini](config/qd.ini).

## Training
python train_ssl.py -config config/qd.ini -model SIMSIAM -gpu 0
  
## Testing
python test_by_search.py -config config/qd.ini -model SIMSIAM -gpu 0

For testing, you should indicate the chekpoint in the configuration file (see CKP_FILE). 


# Dependencies
* https://github.com/jmsaavedrar/datasets
