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



def run_cv(
    model_class: BaseModel,
    train_data: UserDayData,
    k: int,
    lrs: List[float],
    parameter_dict: Dict[str, float]
)->Dict[str, float]:

    num_val_samples = (71 * parameter_dict['cv']) // k
    metrics_by_lr = defaultdict(list)

    for i in range(k):
        val_days = list(range(i * num_val_samples, (i + 1) * num_val_samples))
        val_fold = train_data.get_subset_for_days(val_days)

        train_days = (
            list(range(i * num_val_samples)) +
            list(range((i + 1) * num_val_samples, num_val_samples * k))
        )
        train_fold = train_data.get_subset_for_days(train_days)
        for lr in lrs:
            # for cross-validation purposes,
            # use current learn_rate instead of user-inputted learn rate
            parameter_dict['learn_rate'] = lr

            model = model_class(
                parameter_config=parameter_dict,
            )

            results = ExperimentUtils.run_single_experiment(
                model, train_fold, val_fold)

            # get the evaluation metric, based on what prediction problem we're doing
            if parameter_dict['output_layer']['loss_type'] == 'regression':
                metrics_by_lr[lr].append(results['mse'])
            elif parameter_dict['output_layer']['loss_type'] == 'classification':
                if len(
                parameter_dict['output_layer']['classification_thresholds']
            ) == 1:
                    metrics_by_lr[lr].append(results['AUC'])
                elif len(
                parameter_dict['classification_thresholds']
            ) > 1:
                    metrics_by_lr[lr].append(results['accuracy'])
                else:
                    raise ValueError('If loss_type is classification, \
                        classification_thresholds in the user-inputted .json \
                        must be a string of at least length 1')


    for lr in lrs:
        metrics_by_lr[str(lr) + '_avg'] = np.mean(metrics_by_lr[lr])

    return metrics_by_lr


def main():
    start = datetime.datetime.now()

    with open('init.json') as file:
        parameter_dict = json.load(file)

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_registry = {
        'individual_model': IndividualModel,
        'global_model': GlobalModel,
        'fed_model': FedModel,
    }
    
    if model_registry.get(parameter_dict.get('model_type')) is None:
        raise KeyError("model_type in .json must be one of: 'individual_model', 'global_model', 'fed_model'")
    else:
        model_class = model_registry[parameter_dict['model_type']]

    k = 3
    #lrs = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    #lrs = [0.00001, 0.00005, 0.005, 0.01, 0.03, 0.05] # an hour on jessica's computer
    # lrs = [0.02, 0.03, 0.04, 0.06] # 30 min on jessica's computer
    # lrs = np.linspace(0,1,50, endpoint = False) # 2.55 hours for global model on (0,1,50) on jessica's computer
    lrs = np.linspace(0,1,25, endpoint = False) # 3.6 hours for individiual model on (0,1,25) on jessica's computer, 2.65 hours for global model + fed model on (0,1,25)
    train_data, test_data = ExperimentUtils.simple_train_test_split(parameter_dict)

    metrics_by_lr = run_cv(model_class, train_data, k, lrs, parameter_dict)

    ExperimentUtils.write_to_json(metrics_by_lr, parameter_dict['output_path'] + '_cv_lr')


    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))

if __name__ == '__main__':
    main()

