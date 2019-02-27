# MassFace: an effecient implementation using triplet loss for face recognition.

## Introduction
This project provide an efficient implementation for deep face recognition using Triplet Loss. When trained on [CASIA-Webface](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and tested on on [LFW](http://vis-www.cs.umass.edu/lfw/),this code can achieve an 98.3% accuracy with softmax pretrain and 98.6% with CosFace pretrain . In particular, this repository includes:
![image](./images/framework.png)
- Samples generation
- Hard exmaples mining
- Online and semi-online training
- Multi-gpu implementation

## Data preprocessing

- Align webface and lfw dataset to ```112x112```([casia-112x112](https://pan.baidu.com/s/1MYNq6pkZJCkpKERC92Ea1A),[lfw-112x112](https://pan.baidu.com/s/1-QASgnuL0FYBpzq3K79Vmw)) using [insightface align method](https://github.com/deepinsight/insightface/blob/master/src/align/align_lfw.py)
- If you want to train this code on your dataset, you just need to keep your dataset having the same directory like webface in which all images of the same identity located in one folder.

## Train
- Pretrain with softmax loss: run ```./train_softmax.sh``` with argument ```NETWORK=mobilenet```
- Train with triplet loss: run ```./train_triplet.sh```
## Test
TODO
