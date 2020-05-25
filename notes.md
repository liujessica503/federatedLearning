5/19/2020
Jack and I trying to figure out why running experiments locally for fed and individual give us way different results than running on great lakes
module load python3.6-anaconda
source activate fedModelSetup (already ran conda create -n fedModelSetup and conda install tensorflow==1.14.0 and conda install scikit-learn and conda install matplotlib)
conda list (shows what's installed)
We think the stuff I installed using pip3 is being used

on own computer, installed conda
conda install tensorflow==1.14.0
conda install matplotlib
conda install scikit-learn
conda install keras (installed 2.3.1)
conda install numpy


on great lakes did 
pip3 install --user scikit-learn==0.22.2.post1
conda install numpy==1.18.1
but pretty sure this wasn't needed

4/7/2020
now that I removed user_383848_data.csv from cv3 and cv4 folders
re-running LR tuning for individual and fed models
(I didn't re-tune global, I'm assuming it will continue to be 0.01 LR is best since it's been that way for all CVs)
Did not tune for binary - but I can


in re-test/cv4 indiv regression  "user": 370273 goes up to 400,000 MSE
390039 starts high then goes down to 2-ish
387904 starts high then goes down to less than 1

honing in on CV2 indiv regression since the best was 0.0 last time
this is at the edge of my grid, so I might need to go even smaller
    "1e-10_avg": 2.2832298493661968e+27,
    "5e-10_avg": 2.2832299084007086e+27,
    "1e-09_avg": 2.2832301704093322e+27,
    "5e-09_avg": 2.2832321166853664e+27,
    "1e-08_avg": 2.2832414196345716e+27,
    "5e-08_avg": 2.2833605812854823e+27,

CV3 indiv regression (tried log rates, then honed in)
    "0.32_avg"  7.26845E+12
    "0.35000000000000003_avg"   1.13799E+13
    "0.28_avg"  1.28445E+13
    "0.18000000000000005_avg"   1.31658E+13
    "0.16000000000000003_avg"   1.3166E+13
    "0.31_avg"  1.39181E+13
    "0.4400000000000001_avg"    1.54784E+13
    "0.52_avg"  1.64287E+13
    "0.15000000000000002_avg"   1.81479E+13
    "0.5700000000000001_avg"    1.81696E+13


CV3 fed regression tuning clients with 0.11 LR
    "100_avg"   1.61E+00
    "55_avg"    1.65E+00
    "50_avg"    1.77E+00
    "45_avg"    1.82E+00
    "20_avg"    1.85E+00
    "40_avg"    1.89E+00
    10_avg  1.90E+00
    "35_avg"    1.96E+00
    "120_avg"   1.98E+00

CV3 fed regression (tried log rates, then honed in)
    "0.11000000000000001_avg"   1.726110516
    "0.13_avg"  1.79246968
    "0.52_avg"  1.801924967
    "0.53_avg"  1.812067145
    "0.09000000000000001_avg"   1.821391864
    "0.6800000000000002_avg"    1.829036362
    "0.6600000000000001_avg"    1.836738133
    "0.6700000000000002_avg"    1.837611572
    "0.6300000000000001_avg"    1.84038162
    "0.5_avg"   1.840615717
    "0.7000000000000002_avg"    1.845624845


CV4 fed regression tuning clients with 0.55 LR
    "70_avg"    1.776432689
    "40_avg"    1.786258663
    "20_avg"    1.818021341
    "90_avg"    1.868396426
    "80_avg"    1.872637084
    "95_avg"    1.882750585

CV4 fed regression (tried log rates, then honed in)
    "0.55_avg"  1.723982932
    "0.5_avg"   1.72890779
    "0.08_avg"  1.758537081
    "0.51_avg"  1.764887654
    "0.5800000000000001_avg"    1.767028472
    "0.53_avg"  1.770436982
    "0.5700000000000001_avg"    1.775810271
    "0.54_avg"  1.78261561
    "0.5900000000000001_avg"    1.800696437
    "0.48000000000000015_avg"   1.803548354
    "0.6000000000000001_avg"    1.804741418

CV4 indiv regression (tried log rates, then honed in)
    "0.5700000000000001_avg"    2.19E+18
    "0.56_avg"  2.79E+18
    "0.54_avg"  4.71E+18
    "0.6100000000000001_avg"    5.66E+18
    "0.2_avg"   7.53E+21
    "0.5800000000000001_avg"    5.02E+22
    "0.6400000000000001_avg"    5.13E+22

3/28/2020
for some reason had to
pip3 uninstall tensorflow
pip3 install tensorflow==1.14.0
pip3 install keras==2.2.5
pip3 install --user keras
pip3 install --user sklearn


### Get tuned individual model for cv3 (currently running), and rerun cv3 results to see if order changes (I really hope it doesn't)
### if order doesn't change, re-make box plots and s.d. graph
### if time: give a graph of accuracy after each epoch for cv4

2/29/2020
    "1e-05_avg" 2.31E+27
    "1.4677992676220705e-05_avg"    2.32E+27
    "2.1544346900318823e-05_avg"    2.34E+27
    "3.1622776601683795e-05_avg"    2.36E+27
    "4.641588833612782e-05_avg" 2.40E+27
    "6.812920690579608e-05_avg" 2.46E+27
    "0.0001_avg"    2.55E+27
    "0.0001467799267622069_avg" 2.69E+27

2/27/2020

changed cv_byLearnRate_1.py cv_byLearnRate_2.py cv_byLearnRate_3.py to use learn rates on log-scale 
Then honed in on range of best learn rates


cv4 Fed regression
    "0.06_avg"  1.593332507
    "0.065_avg" 1.662205966
    "0.059_avg" 1.664102478
    "0.066_avg" 1.669555763
    "0.078_avg" 1.684964716
    "0.051_avg" 1.688409977
    "0.063_avg" 1.691069674
    "0.049_avg" 1.69478137

cv4 indiv regression
    "0.71_avg"  5.74E+22
    "0.2_avg"   6.80E+22
    "0.78_avg"  6.31E+23
    "0.28_avg"  1.11E+24
    "0.72_avg"  1.99E+24
    "0.25_avg"  3.27E+24
    "0.4_avg"   3.78E+24
     "0.58_avg" 5.73E+24
    "1e-05_avg" 8.18E+24
    "1.4677992676220705e-05_avg"    8.21E+24
    "2.1544346900318823e-05_avg"    8.26E+24

cv3 Fed regression
    "0.071_avg" 1.625567941
    "0.084_avg" 1.649188858
    "0.074_avg" 1.66323997
    "0.08_avg"  1.669484354
    "0.081_avg" 1.670261456
    "0.076_avg" 1.671497397
    "0.08_avg"  1.676367834

cv3 indiv regression
    "0.14865088937534013_avg"   1.12307E+13
    "0.2227246795350848_avg"    1.3482E+13
        "0.125_avg" 1.99623E+13
    "0.1324328867949119_avg"    2.05142E+13
    "0.1403077560386716_avg"    2.05701E+13
    "0.39685026299204984_avg"   2.11611E+13


2/26/2020

using cv4_ImputeUsingTrainingMean_NoHighIndivMSE (removed the 5 same users as in cv3)

Fed Regression Clients using 0.06 LR
    "50_avg"  1.409194979
    "95_avg"  1.489663302
    "40_avg"  1.518574191
    "45_avg"  1.523261547
    "75_avg"  1.52397777
    "65_avg"  1.62580171
    "30_avg"  1.640123821

Global Regression MSE
    "0.01_avg"  7.32E-01
    "0.02_avg"  7.77E-01
    "0.04_avg"  7.99E-01
    "0.05_avg"  8.33E-01
    "0.07_avg"  8.48E-01
    "0.03_avg"  9.48E-01
    "0.06_avg"  1.19E+00
    "0.08_avg"  2.13E+00
    "0.18_avg"  2.70E+00

Fed Regression MSE
    "0.06_avg"  1.559959648
    "0.05_avg"  1.705897753
    "0.14_avg"  1.754449772
    "0.04_avg"  1.766243647
    "0.03_avg"  1.801159312
    "0.51_avg"  1.812577831
    "0.49_avg"  1.824608365
    "0.1_avg" 1.829592844

Indiv Regression MSE
    "0.2_avg" 6.80E+22
    "0.56_avg"  1.01E+23
    "0.79_avg"  2.90E+23
    "0.78_avg"  5.87E+23
    "0.98_avg"  9.43E+23
    "0.72_avg"  1.27E+24
    "0.65_avg"  1.31E+24

Global Binary AUC
    "0.01_avg"  0.945
    "0.02_avg"  0.943
    "0.03_avg"  0.937
    "0.04_avg"  0.883
    "0.05_avg"  0.881
    "0.08_avg"  0.864
    "0.06_avg"  0.856
    "0.1_avg" 0.782

Fed Binary AUC
    "0.25_avg"  0.932
    "0.21_avg"  0.930
    "0.14_avg"  0.930
    "0.23_avg"  0.930
    "0.26_avg"  0.929
    "0.16_avg"  0.929
    "0.3_avg" 0.929
    "0.17_avg"  0.929

Indiv Binary AUC
    "0.01_avg"  0.87
    "0.09_avg"  0.86
    "0.03_avg"  0.86
    "0.08_avg"  0.86
    "0.02_avg"  0.86
    "0.13_avg"  0.86
    "0.07_avg"  0.86
    "0.15_avg"  0.86

using cv3_ImputeUsingTrainingMean_NoHighIndivMSE (removed the 5 same users as in cv3)

Global regression MSE
    "0.01_avg"  7.91E-01
    "0.02_avg"  8.17E-01
    "0.03_avg"  8.22E-01
    "0.04_avg"  8.52E-01
    "0.05_avg"  9.14E-01
    "0.06_avg"  9.48E-01
    "0.07_avg"  1.74E+00
    "0.08_avg"  2.01E+00
    "0.11_avg"  2.09E+00

Fed regression MSE
    "0.07_avg"  1.619496016
    "0.08_avg"  1.6577071
    "0.12_avg"  1.69261589
    "0.4_avg" 1.821779526
    "0.43_avg"  1.82852594
    "0.45_avg"  1.829050426
    "0.37_avg"  1.835144317

Indiv regression MSE
    "0.31_avg"  6.86E+12
    "0.15_avg"  7.94E+12
    "0.49_avg"  9.04E+12
    "0.28_avg"  1.24E+13
    "0.18_avg"  1.39E+13
    "0.45_avg"  1.50E+13
    "0.39_avg"  1.56E+13
    "0.1_avg" 1.59E+13



Fed Binary Clients using 

Global AUC
    "0.01_avg"  0.937
    "0.02_avg"  0.934
    "0.03_avg"  0.925
    "0.04_avg"  0.911
    "0.05_avg"  0.878
    "0.06_avg"  0.823
    "0.14_avg"  0.819

Fed AUC
    "0.29_avg"  0.919
    "0.26_avg"  0.917
    "0.31_avg"  0.916
    "0.15_avg"  0.914
    "0.1_avg" 0.914
    "0.22_avg"  0.914

Indiv AUC
    "0.01_avg"  0.861
    "0.02_avg"  0.857
    "0.03_avg"  0.855
    "0.04_avg"  0.854
    "0.13_avg"  0.852
    "0.1_avg" 0.851



2/20/2020

using cv3_ImputeUsingTrainingMean_NoHighIndivMSE

Fed Binary clients using 0.31 learn rate
    "95_avg"  0.896815519
    "60_avg"  0.895468549
    "100_avg" 0.894617256
    "90_avg"  0.894498165
    "85_avg"  0.894367821

Global AUC
    "0.01_avg"  0.921998573
    "0.02_avg"  0.92006215
    "0.03_avg"  0.887491506
    "0.06_avg"  0.817831818
    "0.04_avg"  0.814658962
    "0.05_avg"  0.727554484
    "0.1_avg" 0.719177009

Indiv AUC
    "0.01_avg"  0.828763379
    "0.02_avg"  0.827081232
    "0.03_avg"  0.824513397
    "0.04_avg"  0.822210149
    "0.05_avg"  0.821529358
    "0.1_avg" 0.816389302
    "0.24_avg"  0.815958649
    "0.06_avg"  0.815400586
    "0.22_avg"  0.815253144

Fed AUC
    "0.31_avg"  0.900391791
    "0.37_avg"  0.896435824
    "0.59_avg"  0.896425155
    "0.33_avg"  0.895557368
    "0.24_avg"  0.895474983
    "0.3_avg" 0.894592918
    "0.2_avg" 0.894254474
    "0.32_avg"  0.893534019

2/11/2020

using cv3_ImputeUsingTrainingMean_NoHighIndivMSE

Indiv Regression MSE
0.0_avg 8.89E+26
0.66_avg  1.99E+27
0.63_avg  2.61E+27
0.33_avg  2.62E+27
0.67_avg  2.63E+27
0.47_avg  2.91E+27
0.79_avg  3.33E+27
0.92_avg  3.76E+27

Global Regression MSE
0.01_avg  0.866447943
0.02_avg  0.951017967
0.03_avg  1.009405698
0.04_avg  1.064829105
0.05_avg  1.125168878
0.07_avg  1.419766274

Fed clients MSE
60_avg  1.578189348
35_avg  1.647603772
70_avg  1.672587978
50_avg  1.715635309
45_avg  1.721569899
80_avg  1.749894049
55_avg  1.786046023

Fed Regression MSE (lower is better)
0.12_avg  1.638582956
0.09_avg  1.651191474
0.1_avg 1.686909434
0.14_avg  1.690706475
0.13_avg  1.69377902
0.11_avg  1.721028404
0.85_avg  1.741138477
0.15_avg  1.749589614
0.22_avg  1.762814028


1/29/2020

to get reproducible results: PYTHONHASHSEED=0 python3 single_experiment.py global_inits.json
from https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras

I think this is CV2

Fed Binary AUC with 80 clients (higher is better)
"0.25_avg"  0.900575287
"0.21_avg"  0.894125669
"0.19_avg"  0.893263471
"0.28_avg"  0.893023748
"0.22_avg"  0.892033463
"0.14_avg"  0.891639286
"0.1_avg" 0.890702894
"0.4_avg" 0.890515416
"0.32_avg"  0.889545411
"0.27_avg"  0.888632238

Global Binary AUC 
"0.01_avg"  0.921491775
"0.02_avg"  0.91992945
"0.03_avg"  0.917196659
"0.04_avg"  0.905406546
"0.05_avg"  0.896677757
"0.06_avg"  0.817326739

Indiv binary AUC
"0.02_avg"  0.826979897
"0.03_avg"  0.825991278
"0.01_avg"  0.824524843
"0.05_avg"  0.821902881
"0.04_avg"  0.820704836
"0.07_avg"  0.818736322
"0.11_avg"  0.816747184

Indiv regression MSE (lower is better)
"0.0_avg" 8.83E+26
"0.71_avg"  2.19E+27
"0.81_avg"  2.67E+27
"0.38_avg"  2.78E+27
"0.33999999999999997_avg" 3.06E+27
"0.4_avg" 3.37E+27
"0.84_avg"  4.23E+27
"0.15_avg"  4.85E+27
"0.63_avg"  4.86E+27

Fed Regression clients per round (with LR 13?)
80_avg  1.536933826
65_avg  1.61759556
35_avg  1.62195922
55_avg  1.633368288
60_avg  1.674157817
    
Fed Regression MSE
0.13_avg  1.844462626
0.95_avg  1.890134584
0.12_avg  1.907395283
0.94_avg  1.941585601
0.08_avg  1.959908989
0.06_avg  1.961902227
0.07_avg  1.9643402
0.09_avg  1.977176007
0.11_avg  1.996468725
0.89_avg  2.003485648

Global regression MSE
"0.01_avg"  0.85559959
"0.02_avg"  0.899105324
"0.03_avg"  0.908518733
"0.05_avg"  0.976826785
"0.04_avg"  0.999085865
"0.06_avg"  1.248831845


imputing using only training set mean (previous runs were imputing using mean over all time)
_____________________________________________________
1/23/2020 Binary AUC (higher is better): 
Fed model 
0.27_avg  0.899162413
0.29_avg  0.898062568
0.21_avg  0.896038984
0.3_avg 0.89417863
0.23_avg  0.893821203
0.22_avg  0.893472015
0.24_avg  0.891958814
0.33_avg  0.891206446
0.31_avg  0.890943948

Global model
0.01_avg  0.921752467
0.02_avg  0.919208222
0.03_avg  0.918744545
0.05_avg  0.888845745
0.04_avg  0.87064254
0.06_avg  0.791502855
0.07_avg  0.772183144
0.08_avg  0.761994426
0.11_avg  0.734036142

Individual model
0.02_avg  0.827060037
0.01_avg  0.824161653
0.04_avg  0.823487657
0.03_avg  0.823487475
0.06_avg  0.821277533
0.05_avg  0.820083034
0.11_avg  0.817721256
0.08_avg  0.817565062
0.09_avg  0.817419716


1/21/2020 Regression MSE (lower is better): 

Tuning fed clients per round, with 2 local updates and lr 0.96
80_avg  1.611718309
70_avg  1.615479005
60_avg  1.723917922
35_avg  1.728839075
55_avg  1.766833311

Fed model: best is 2 local updates (as opposed to 1), best lrs
0.96_avg  1.834488905
0.97_avg  1.889867852
0.98_avg  1.93033621
0.9299999999999999_avg  1.932946893
0.14_avg  1.98336081
0.71_avg  2.012272368
0.11_avg  2.022504984
0.13_avg  2.036682907
0.75_avg  2.044841517

Global model:
    "0.01_avg"  0.864588442
    "0.04_avg"  0.918783291
    "0.03_avg"  0.920699781
    "0.02_avg"  0.92161917
    "0.05_avg"  1.084013267
    "0.07_avg"  1.429105268
    "0.06_avg"  1.453938829
    "0.08_avg"  1.752918575
    "0.09_avg"  1.905206922

Individual model (including high MAE files):
0.0_avg  8.83E+26
0.83_avg  1.23E+27
0.5_avg 2.49E+27
0.88_avg  3.33E+27
0.96_avg  4.34E+27
0.47_avg  5.77E+27
0.24_avg  5.80E+27
0.37_avg  7.33E+27
0.91_avg  8.86E+27
0.54_avg  8.95E+27

individual model without high MAE:
    "0.6799999999999999_avg"  8.62E+26
    "0.0_avg" 8.89E+26
    "0.81_avg"  2.77E+27
    "0.5_avg" 3.07E+27
    "0.6_avg" 3.90E+27
    "0.4_avg" 6.70E+27
    "0.41000000000000003_avg" 9.57E+27
    "0.45999999999999996_avg" 9.82E+27

# Federated Learning
Work on the feasibility of applying federated learning to IHS data


cv3_indiv_regression.csv_by_user_(individual_model).json results has 4 users that have 10^29 or 10^30 MSE
removing those 4 users, avg MSE is 7.2
Some of these users have really low MSE in TensorFlow_Binary_draft_0.64lr.csv file, others have high MSE

12/20/2019
Trying to run cv on regression
After Jack and I implemented regression + multiclass

run_single_experiment.py is now gone, using ExperimentUtils for run_single_experiment, simple_train_test_split, write_to_json etc

if "loss_type": "regression", then "classification_thresholds": "" should be empty and plot_auc should be 0. Don't need prediction classes since in regression we wouldn't be binning any classes together.
if "loss_type": "classification" then "classification_thresholds": [7] or whatever list of thresholds you want like   "classification_thresholds": [3,7] for multi-class

Should only one of classification_thresholds and prediction_classes occur?? Should they be the same? (01/23/20 removed prediction_classes and ran binary predictions on great lakes)

prediction_classes is only used if loss type == classification

cv3_indiv_regression.csv_cv_lr.json last metric is WRONG in each, should not be greater than 10 because it's mean absolute error...
  potential fixes? OutputLayer.py self.loss = "mean_squared_error" changed to self.loss = "mse" but I don't think this is the issue since only indiv_regression metrics seem really off
       
  cv_byLearnRate was using init.json's learn_rate, and we weren't overwriting it. Changed from parameter_overwrite={'lr': lr} to parameter_overwrite={'learn_rate': lr} to fix this.

for indiv models, taking average of k=0 and k=1 (excluding k=2) still gives best is 0.64


for greatlakes, removing 'epsilon=None' in model.compile fixed the 'None values not supported' error.
in greatlakes -- i need tensorflow 1.14.0 not the higher 1.15 version, and need msgpack for tensorflow. I also installed tensorflow==1.14 but then I also installed tensorflow-gpu==1.14:

# try running tensorflow binary draft step by step, send error to hpc support; ask on slack if anyone has used tensorflow on greatlakes

# trying this 1/17/20 https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc
module load python3.6-anaconda/5.2.0
conda create --name tf_gpu tensorflow-gpu==1.14.0
# To activate this environment, use:
# > source activate tf_gpu
#
# To deactivate an active environment, use:
# > source deactivate
#
source activate tf_gpu (had to exit and try this a few times)
pip install --user keras
pip install --user sklearn
pip install --user matplotlib
pip install --user pandas

in greatlakes, cd federatedLearning
sbatch global_cv.sbat
see powerpoint slides in gmail google drive

https://arc-ts.umich.edu/greatlakes/software/tensorflow/