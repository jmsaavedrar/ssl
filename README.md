# SSL:  A repository for self-supervised learning

This repository contains implementations of SimSiam and BYOL, which were tested with the quickdraw dataset. 
To make the data loading easy, we use TFDS. TFDS allows us to load popular datasets as ImageNet or build our own datasets. In our case, we use our own dataset [tfds_qd](https://github.com/jmsaavedrar/datasets/tree/main/tfds_qd).

In addition, we add configuration file that may facilitate experimentation.

## Training
train_ssl -config config/qd.ini -model SIMSIAM
  
## Testing
test_by_search -config config/qd.ini -model SIMSIAM



# QuicDraw Dataset
We have prepared a subset of QuickDraw with 1100 instances per class. You can download it from [here](https://www.dropbox.com/s/eq3vzu65elii62i/tfds_qd.tar).

# Dependencies
* https://github.com/jmsaavedrar/datasets
