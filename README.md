# stopro
libraries for implementing Physics-Informed Gaussian Process Regression mainly on fluid problems

# Requirements
GPU
- cuda: 11.x (5~)
- cudnn: 8.x.x (3.3~)
- nccl: 2.x.x  
- 
Python
- python=3.11
- Summarized in requirements.txt

# MEMO
## confirm algorithm for implementing BBMM

1. preapare data
2. setup func to calc loss and gradient of loss
   1. setup func to calculate covariance matrix
   2. setup func to calculate matrix-matri multiplication
3. training
   1. calculate gradient of loss and loss to update hyperparameters
   2. check convergence
4. prediction
   1. calculate mean and variance of prediction

## what to develop
- algorithm to calculate loss and gradient of loss
- algorithm to calculate mean and variance of prediction
  
## Todo
- implement prediction
  - calc $K_{XX}^{-1}\boldsymbol{y}, K_{XX}^{-1}\boldsymbol{k}_{Xx^*}$ by bbmm
- implement loss and gradient of loss
  - calc the log deterninant for loss
  - calc the trace term for gradient of loss
  
## Installation
Clone this repository to the location where your Python can refer.
In this example, we assume that this repository is clone to the directory `$HOME/opt/`.
```bash
cd ~/opt
git clone https://github.com/ogaken1104/stopro.git
```




