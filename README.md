# stopro
libraries for implementing Physics-Informed Gaussian Process Regression mainly on fluid problems

## Requirements
GPU
- cuda: 11.x (5~)
- cudnn: 8.x.x (3.3~)
- nccl: 2.x.x  
- 
Python
- python=3.11
- Summarized in `requirements.txt`

  
## Installation
Clone this repository to the location where your Python can refer.
In this example, we assume that this repository is clone to the directory `$HOME/opt/`.
```bash
cd ~/opt
git clone https://github.com/ogaken1104/stopro.git
```
Please make sure your python can refer to the directory `~/opt/stopro`, by adding below script to your .bashrc or .zshrc.
```bash
export PYTHONPATH=$PYTHONPATH:~/opt/stopro
```

## Structure of Program
We summarize the structure of the program. Details are described in docstrings of each script.
### Modules for GP calculations
`GP`
- GP classes for handling all calculations about GP.
- `gp.py`: base class for all GP.
- `gp_2D.py`, `gp_3D.py`: base class for 2D and 3D GP. `gp_3D.py` inherits `gp_2D.py`.

`solver`
- optimization scheme.

`sub_modules`

### Other modules
`data_handler`
- `data_handle_module.py`: class for loading and saving data.

`data_generator`
- classes for generating data for each system

`data_preparer`
- `data_preparer.py`: class for preparing data, using `data_generator`, `data_handler`

`analyzer`
- `analysis.py`: common functions for analyzing calculation results
- `plot_xxx.py`: plot functions for each system

### Data
`default_params`: 
- `params_prepare.yaml`: settings for preparing data
- `params_main.yaml`: settings for main calculation
- `lbls.yaml`: common labels for data

`template_data`:
- FEM or SPM references

## Test
```bash
cd ~/opt/stopro
pytest ./test
```
By running the above command, you can test for
- flow between sinusoidal walls (forward problem)
- sin 1D naive
- Poiseuille flow
- sin 1D with Laplasian
- gaussian for 3D
- drag flow in 3D
- flow between sinusoidal walls (using explicit derivative of loss)

In `tests_develop`, tests code under development are stored.



<!-- # MEMO
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
  - calc the trace term for gradient of loss -->