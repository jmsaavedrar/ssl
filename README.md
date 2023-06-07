# SSL:  A repository for self-supervised learning

This repository contains implementations of SimSiam and BYOL, which were tested with the quickdraw dataset. 
To make the data loading easy, we use TFDS. TFDS allows us to load popular datasets as ImageNet or build our own datasets. In our case, we use our own dataset [tfds_qd](https://github.com/jmsaavedrar/datasets/tree/main/tfds_qd).

In addition, we add configuration file that may facilitate experimentation.

## Training
train_ssl -config config/qd.ini -model SIMSIAM
  
## Testing
test_by_search -config config/qd.ini -model SIMSIAM




