# Federated Learning on Mobile Health Data
This repository contains the code to reproduce the results in the following publication:
Liu JC, Goetz J, Sen S, Tewari A. Learning From Others Without Sacrificing Privacy: Simulation Comparing Centralized and Federated Machine Learning on Mobile Health Data. JMIR Mhealth Uhealth 2021;9(3):e23728. doi: 10.2196/23728. Available at https://mhealth.jmir.org/2021/3/e23728/


## Setup
To replicate results, please install Python 3.7.7 and the following libraries and versions:

* tensorflow==1.14.0
* python==3.7.7
* scikit-learn==0.22.1
* scipy==1.4.1 
* pandas=1.0.1
* keras=2.3.1
* numpy==1.18.1
* matplotlib==3.1.3 

The full list of specs is in spec-file.txt

To create an identical environment, enter on the command line:
`conda create --name myenv --file spec-file.txt`
`conda activate myenv`

For more info: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#cloning-an-environment


## Running a single experiment: 
1. Clone this repository, and make sure you are in the directory that contains the repository. 
2. Enter the following on the command line:
`PYTHONHASHSEED=123456 python3 single_experiment.py fed_init_wesad.json`
3. Change the last argument in 2. to be the file you want to run (see next paragraph for the corresponding file and model). 

Files corresponding to each model: 
* Individual model: indiv_init_wesad.json
* Server Personalized Model: global_pers_init_wesad.json
* Federated Personalized Model: fed_pers_init_wesad.json
* Server (non-personalized) Model: global_init_wesad.json
* Federated (non-personalized) Model: fed_init_wesad.json

seed 1234 is used in each of the above .json files. The seeds used for the box plot in the paper appendix are:
1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1247 1248


## Sources
Data was transformed from [1], using scripts derived from https://github.com/WJMatthew/WESAD#matthew-johnson-2019  

[1] Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction;2018:400-408. New York: Association for Computing Machinery. doi:10.1145/3242969.3242985 
WESAD Dataset available at https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29

