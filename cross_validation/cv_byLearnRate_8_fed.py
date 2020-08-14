# outputs a nested dictionary with the following keys, in order:
# epoch
# clients per round
# local updates
# fed_stepsize
# learning rate: AUC or mse

import json
import sys
import numpy as np
import datetime
from operator import itemgetter
# seeing if this will fix memory leak
from keras import backend as K
import tensorflow as tf

# import user-defined functions

from IndividualModel import IndividualModel
from GlobalModel import GlobalModel
from GlobalModelPersonalized import GlobalModelPersonalized
from FedModel import FedModel
from FedModelPersonalized import FedModelPersonalized
from MovingMeanModel import MovingMeanModel
from BaselineModel import BaselineModel
from BaseModel import BaseModel
from UserDayData import UserDayData
from collections import defaultdict
from ExperimentUtils import ExperimentUtils
from typing import List, Dict

# standardize the data
from sklearn.preprocessing import StandardScaler
import pandas as pd


def run_cv(
    model_class: BaseModel,
    train_data: UserDayData,
    k: int,
    epochs: List[int],
    clients_per_round_list: List[int], 
    local_updates_per_round_list: List[int], 
    fed_stepsize_list: List[float], 
    lrs: List[float],
    parameter_dict: Dict[str, float],
    user_list: List[int]
)->Dict[str, float]:

    num_val_samples = 50 // k


    # save metrics in dictionary
    metrics_by_lr = defaultdict(list)

    for epoch in epochs:
        metrics_by_lr['Epoch' + '_' + str(epoch)] = defaultdict(list)
        # for cross-validation purposes,
        # use current epoch instead of user-inputted epoch
        parameter_dict['epochs'] = epoch

        for clients_per_round in clients_per_round_list:
            metrics_by_lr[
            'Epoch_' + str(epoch)]['Clients_' + str(clients_per_round)
            ] = defaultdict(list)
            parameter_dict['fed_model_parameters']['clients_per_round'] = clients_per_round

            for local_update in local_updates_per_round_list:
                metrics_by_lr[
                'Epoch_' + str(epoch)]['Clients_' + str(clients_per_round)][
                'Local_Updates_' + str(local_update)
                ] = defaultdict(list)
                parameter_dict['fed_model_parameters']['local_updates_per_round'] = local_update

                for fed_stepsize in fed_stepsize_list:
                    metrics_by_lr[
                    'Epoch_' + str(epoch)]['Clients_' + str(clients_per_round)][
                    'Local_Updates_' + str(local_update)][
                    'Fed_stepsize_' + str(fed_stepsize)
                    ] = defaultdict(list)
                    parameter_dict['fed_model_parameters']['fed_stepsize'] = fed_stepsize

                    for lr in lrs:
                        # for cross-validation purposes,
                        # use current learn_rate instead of user-inputted learn rate

                        parameter_dict['learn_rate'] = lr

                        for i in range(k):


                            val_pairs = []
                            train_pairs = []
                            for user in user_list:
                                # we need to set the validation days for each user separately
                                # because the order of tasks differed across users
                                # and number of measurements differed very slightly 
                                # for each task for each user

                                # get the indices where the classification label changes
                                # this only works if the first and last values are NOT the same, 
                                # which in our case is true, because the labeled tasks
                                # (baseline, stress, amusement)
                                # occur sequentially and do not repeat

                                # get the classes in the validation set for each user

                                user_data = train_data.get_subset_for_users([user])
                                user_days = [x[1] for x in user_data.user_day_pairs]

                                user_val_y = user_data.get_y()
                                # 7/15/2020 choosing 1 data point from each class for training
                                #user_train_y = user_data.get_y()

                    
                                '''
                                # index where class a, class b, class c start
                                class_b_start = np.where(np.roll(user_train_y,1)!=user_train_y)[0][1]
                                class_c_start = np.where(np.roll(user_train_y,1)!=user_train_y)[0][2]

                                class_a_idx = np.random.randint(low = 0, high = class_b_start)
                                class_b_idx = np.random.randint(low = class_b_start, high = class_c_start)
                                class_c_idx = np.random.randint(low = class_c_start, high = len(user_train_y))

                                user_train_idx = [class_a_idx, class_b_idx, class_c_idx]
                                # get the days we want for training, based on our selected indices
                                mask = np.zeros(np.array(user_days).shape,dtype = bool)
                                mask[user_train_idx] = True
                                user_train_days = np.array(user_days)[mask]

                                for x in user_train_days:
                                    train_pairs.append((user, x))

                                # use the rest of the training data for validation set
                                user_val_days = np.array(user_days)[~mask]
                                for x in user_val_days:
                                    val_pairs.append((user, x))
                                '''

                                # 7/15/2020 comment out if we're choosing 1 data point from each class for training
                                # index where class a, class b, class c start
                                class_b_start = np.where(np.roll(user_val_y,1)!=user_val_y)[0][1]
                                class_c_start = np.where(np.roll(user_val_y,1)!=user_val_y)[0][2]

                                # get indices of user_days that we want for validation set
                                class_a_idx = list(range(class_b_start // 3 * i, class_b_start // 3 * (i + 1)))
                                class_b_idx = list(range(class_b_start + (class_c_start - class_b_start) // 3 * i, class_b_start + (class_c_start - class_b_start)//3 * (i + 1)))
                                class_c_idx = list(range(class_c_start + (len(user_val_y) - class_c_start) // 3 * i, class_c_start + (len(user_val_y) - class_c_start) // 3 * (i + 1)))

                                user_val_idx = class_a_idx + class_b_idx + class_c_idx

                                # get the days we want for validation, based on our selected indices
                                mask = np.zeros(np.array(user_days).shape,dtype = bool)
                                mask[user_val_idx] = True
                                user_val_days = np.array(user_days)[mask]

                                for x in user_val_days:
                                    val_pairs.append((user, x))

                                # use the rest of the training data for training set
                                user_train_days = np.array(user_days)[~mask]
                                for x in user_train_days:
                                    train_pairs.append((user, x))
                                

                            #train_fold = train_data.get_subset_for_user_day_pairs(train_pairs)
                            #val_fold = train_data.get_subset_for_user_day_pairs(val_pairs)

                            
                            # just to test, using global model's standardization 
                            tmp_train_fold = train_data.get_subset_for_user_day_pairs(train_pairs)
                            tmp_val_fold = train_data.get_subset_for_user_day_pairs(val_pairs)
                            # end
                            

                            # seeing if this will fix memory leak
                            K.clear_session()

                            session_conf = tf.ConfigProto(
                                intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
                            )
                            sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
                            K.set_session(sess)                             

                            model = model_class(
                                parameter_config=parameter_dict,
                            )


                            # standardization for fed models
                            scaler = StandardScaler().fit(tmp_train_fold.get_X())
                            train_fold_X_np = scaler.transform(tmp_train_fold.get_X())
                            train_fold_X = pd.DataFrame(train_fold_X_np, columns = tmp_train_fold.get_X().columns)
                            val_fold_X_np = scaler.transform(tmp_val_fold.get_X())
                            val_fold_X = pd.DataFrame(val_fold_X_np, columns = tmp_val_fold.get_X().columns)
                            train_fold = UserDayData(X=train_fold_X, user_day_pairs = tmp_train_fold.get_user_day_pairs(), y = tmp_train_fold.get_y())
                            val_fold = UserDayData(X=val_fold_X, user_day_pairs = tmp_val_fold.get_user_day_pairs(), y = tmp_val_fold.get_y())
                            # end


                            results = ExperimentUtils.run_single_experiment(
                                model, train_fold, val_fold)

                            # get the evaluation metric, based on what prediction problem we're doing
                            if parameter_dict['output_layer']['loss_type'] == 'regression':
                                metrics_by_lr[
                                'Epoch_' + str(epoch)][
                                'Clients_' + str(clients_per_round)][
                                'Local_Updates_' + str(local_update)][
                                'Fed_stepsize_' + str(fed_stepsize)][
                                'Learn_rate_' + str(lr)].append(results['mse'])
                            elif parameter_dict['output_layer']['loss_type'] == 'classification':
                                if len(
                                parameter_dict['output_layer']['classification_thresholds']
                            ) == 1:
                                    metrics_by_lr[
                                    'Epoch_' + str(epoch)][
                                    'Clients_' + str(clients_per_round)][
                                    'Local_Updates_' + str(local_update)][
                                    'Fed_stepsize_' + str(fed_stepsize)][
                                    'Learn_rate_' + str(lr)].append(results['AUC'])
                                elif len(
                                parameter_dict['output_layer']['classification_thresholds']
                            ) > 1:
                                    metrics_by_lr[
                                    'Epoch_' + str(epoch)][
                                    'Clients_' + str(clients_per_round)][
                                    'Local_Updates_' + str(local_update)][
                                    'Fed_stepsize_' + str(fed_stepsize)][
                                    'Learn_rate_' + str(lr)].append(results['accuracy'])
                                else:
                                    raise ValueError('If loss_type is classification, \
                                        classification_thresholds in the user-inputted .json \
                                        must have at least length 1')
                                    
                            # write to file
                            #ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + "_(" + parameter_dict['model_type'] + ")" + 'tmp_cv_lr')
                            output_path = parameter_dict['output_path'] + "_(" + parameter_dict['model_type'] + ")" + 'tmp_cv_lr'
                            with open(output_path + ".json", "w") as f:
                                 json.dump(metrics_by_lr, f, indent=4)
            
    for epoch in epochs:
        metrics_by_lr['Epoch_' + str(epoch) + '_avg'] = defaultdict(list)
        for clients_per_round in clients_per_round_list:
            metrics_by_lr['Epoch_' + str(epoch) + '_avg'][
            'Clients_' + str(clients_per_round) + '_avg'] = defaultdict(list)
            for local_update in local_updates_per_round_list:
                metrics_by_lr['Epoch_' + str(epoch) + '_avg'][
                'Clients_' + str(clients_per_round) + '_avg'][
                'Local_Updates_' + str(local_update) + '_avg'] = defaultdict(list)
                for fed_stepsize in fed_stepsize_list:
                    metrics_by_lr['Epoch_' + str(epoch) + '_avg'][
                    'Clients_' + str(clients_per_round) + '_avg'][
                    'Local_Updates_' + str(local_update) + '_avg'][
                    'Fed_stepsize_' + str(fed_stepsize) + '_avg'] = defaultdict(list)
                    for lr in lrs:
                        metrics_by_lr['Epoch_' + str(epoch) + '_avg'][
                        'Clients_' + str(clients_per_round) + '_avg'][
                        'Local_Updates_' + str(local_update) + '_avg'][
                        'Fed_stepsize_' + str(fed_stepsize) + '_avg'][
                        'Learn_rate_' + str(lr) + '_avg'
                        ] = np.mean(metrics_by_lr[
                            'Epoch_' + str(epoch)][
                            'Clients_' + str(clients_per_round)][
                            'Local_Updates_' + str(local_update)][
                            'Fed_stepsize_' + str(fed_stepsize)][
                            'Learn_rate_' + str(lr)
                        ])

    return metrics_by_lr


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_registry = {
        "individual_model": IndividualModel,
        "global_model": GlobalModel,
        "global_model_pers": GlobalModelPersonalized,
        "fed_model": FedModel,
        "fed_model_pers": FedModelPersonalized,
        "moving_mean_model": MovingMeanModel,
        "baseline_model": BaselineModel,
    }
    
    if model_registry.get(parameter_dict.get('model_type')) is None:
        raise KeyError(
                'model_type in config json must be one of: "individual_model",'
                '"global_model", "fed_model", "fed_model_pers", "global_model_pers", "moving_mean_model", "baseline_model"' 
            )
    else:
        model_class = model_registry[parameter_dict['model_type']]

    k = 3

    # 5, 10, 25, 50 % of users
    #clients_per_round_list = [20, 40, 100, 200]
    clients_per_round_list = [1, 2, 4, 8, 10, 15]

    #local_updates_per_round_list = [2, 4, 6, 8]
    #local_updates_per_round_list = [8000, 10000, 50000]
    local_updates_per_round_list = [1, 2, 4, 8, 12]

    #fed_stepsize_list = [1]
    fed_stepsize_list = [0.01, 0.1, 0.5, 0.7, 1]

    #lrs = [0.00001, 0.00005, 0.005, 0.01, 0.03, 0.05] # an hour on jessica's computer
    # lrs = [0.02, 0.03, 0.04, 0.06] # 30 min on jessica's computer
    # lrs = np.linspace(0,1,50, endpoint = False) # 2.55 hours for global model on (0,1,50) on jessica's computer
    # lrs = np.linspace(0,1,25, endpoint = False) # 3.6 hours for individiual model on (0,1,25) on jessica's computer, 2.65 hours for global model + fed model on (0,1,25)
    # lrs = np.linspace(0,0.25,25, endpoint = False)
    #lrs = [1e-10, 1e-08, 1e-06, 1e-05, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]
    #lrs = np.arange(0.01, 0.05, 0.01)
    #lrs = np.logspace(-5, -1, base = 10, num = 25)
    #lrs = [0.002, 0.004, 0.006, 0.008, 0.01]
    #lrs = np.concatenate([np.arange(0.005,0.01,0.001), np.arange(0.01,0.05,0.01)])
    #lrs = np.arange(0.01, 0.1,0.01)
    lrs = [0.7, 0.9]

    # tune number of epochs jointly with learning rates
    #epochs = np.arange(10,80,20)
    #epochs = np.arange(30,80,20)
    #epochs = np.concatenate([np.arange(5,25,5), [40,50,60]])
    epochs = [1, 2, 5, 10, 20, 30, 40]

    user_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    #train_data, test_data = ExperimentUtils.raw_train_test_split(parameter_dict)
    train_data, test_data = ExperimentUtils.simple_train_test_split(parameter_dict)


    metrics_by_lr = run_cv(model_class, train_data, k, epochs, clients_per_round_list, local_updates_per_round_list, fed_stepsize_list, lrs, parameter_dict, user_list)

    ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + "_(" + parameter_dict['model_type'] + ")" + '_cv_lr')


    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))

if __name__ == '__main__':
    main()
