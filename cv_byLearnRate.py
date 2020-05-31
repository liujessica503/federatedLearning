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


def run_cv(
    model_class: BaseModel,
    train_data: UserDayData,
    k: int,
    epochs: List[int],
    clients_per_round_list: List[int], 
    local_updates_per_round_list: List[int], 
    fed_stepsize_list: List[float], 
    lrs: List[float],
    parameter_dict: Dict[str, float]
)->Dict[str, float]:

    num_val_samples = (71 * parameter_dict['cv']) // k

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
                            val_days = list(range(i * num_val_samples, (i + 1) * num_val_samples))
                            val_fold = train_data.get_subset_for_days(val_days)

                            train_days = (
                                list(range(i * num_val_samples)) +
                                list(range((i + 1) * num_val_samples, num_val_samples * k))
                            )
                            train_fold = train_data.get_subset_for_days(train_days)

                            model = model_class(
                                parameter_config=parameter_dict,
                            )

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
                                parameter_dict['classification_thresholds']
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
                            #ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + 'tmp_cv_lr')
                            output_path = parameter_dict['output_path'] + 'tmp_cv_lr'
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

    # change 5, 10, 25, 50 % of users
    clients_per_round_list = [20, 40, 100, 200]

    local_updates_per_round_list = [1, 2, 4, 6, 8, 10, 12]

    fed_stepsize_list = [1e-05, 1e-03, 0.1, 0.2, 0.5, 1]

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
    lrs = [1e-10, 1e-08, 1e-06, 1e-05]

    # tune number of epochs jointly with learning rates
    epochs = np.arange(10,80,20)
    #epochs = np.concatenate([np.arange(5,25,5), [40,50,60]])
    #epochs = [40, 50]


    train_data, test_data = ExperimentUtils.simple_train_test_split(parameter_dict)

    metrics_by_lr = run_cv(model_class, train_data, k, epochs, clients_per_round_list, local_updates_per_round_list, fed_stepsize_list, lrs, parameter_dict)

    ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + '_cv_lr')


    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))

if __name__ == '__main__':
    main()

