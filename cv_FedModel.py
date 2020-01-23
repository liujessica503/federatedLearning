import json
import sys
import numpy as np
import datetime

# import user-defined functions

from BaseModel import BaseModel
from GlobalModel import GlobalModel
from IndividualModel import IndividualModel
from FedModel import FedModel
from UserDayData import UserDayData
from collections import defaultdict
from ExperimentUtils import ExperimentUtils
from typing import List, Dict
import numpy as np


def run_cv_FedModel_clients(
    model_class: BaseModel,
    train_data: UserDayData,
    k: int,
    clients_per_round_list: List[float],
    parameter_dict: Dict[str, float]
)->Dict[str, float]:

    num_val_samples = (71 * parameter_dict['cv']) // k
    metrics_by_clients = defaultdict(list)

    for i in range(k):
        val_days = list(range(i * num_val_samples, (i + 1) * num_val_samples))
        val_fold = train_data.get_subset_for_days(val_days)

        train_days = (
            list(range(i * num_val_samples)) +
            list(range((i + 1) * num_val_samples, num_val_samples * k))
        )
        train_fold = train_data.get_subset_for_days(train_days)

        for clients_per_round in clients_per_round_list:
            # for cross-validation purposes,
            # use current learn_rate instead of user-inputted learn rate
            parameter_dict['fed_model_parameters']['clients_per_round'] = clients_per_round

            model = model_class(
                parameter_config=parameter_dict,
            )

            results = ExperimentUtils.run_single_experiment(
                model, train_fold, val_fold)

            # get the evaluation metric, based on what prediction problem we're doing
            if parameter_dict['output_layer']['loss_type'] == 'regression':
                metrics_by_clients[clients_per_round].append(results['mse'])
            elif parameter_dict['output_layer']['loss_type'] == 'classification':
                if len(
                parameter_dict['output_layer']['classification_thresholds']
            ) == 1:
                    metrics_by_clients[clients_per_round].append(results['AUC'])
                elif len(
                parameter_dict['classification_thresholds']
            ) > 1:
                    metrics_by_clients[clients_per_round].append(results['accuracy'])
                else:
                    raise ValueError('If loss_type is classification, \
                        classification_thresholds in the user-inputted .json \
                        must be a string of at least length 1')
                    
            # write progress to file
            ExperimentUtils.write_to_json(metrics_by_clients, 'tmp_output' + '_cv_clients')

    for clients_per_round in clients_per_round_list:
        metrics_by_clients[str(clients_per_round) + '_avg'] = np.mean(metrics_by_clients[clients_per_round])

    return metrics_by_clients


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_registry = {
        "fed_model": FedModel,
    }
    
    if model_registry.get(parameter_dict.get('model_type')) is None:
        raise KeyError('model_type in .json must be "fed_model"')
    else:
        model_class = model_registry[parameter_dict['model_type']]

    k = 3
    
    clients_per_round_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

    train_data, test_data = ExperimentUtils.simple_train_test_split(parameter_dict)

    metrics_by_clients = run_cv_FedModel_clients(model_class, train_data, k, clients_per_round_list, parameter_dict)

    ExperimentUtils.write_to_json(metrics_by_clients, parameter_dict['output_path'] + '_cv_clients')


    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))

if __name__ == '__main__':
    main()

