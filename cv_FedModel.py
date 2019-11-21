import json
import sys
import numpy as np

# import user-defined functions
from BaseModel import BaseModel
from FedModel import FedModel
from UserDayData import UserDayData
from collections import defaultdict
from single_global_experiment import (
    run_single_experiment,
    write_to_json,
    simple_train_test_split,
)
from typing import List, Dict


def run_cv_FedModel(
    model_class: BaseModel,
    train_data: UserDayData,
    k: int,
    clients_per_round_list: List[float],
    parameter_dict: Dict[str, float]
)->Dict[str, float]:

    num_val_samples = (71 * parameter_dict["cv"]) // k
    aucs_by_clients = defaultdict(list)

    for i in range(k):
        val_days = list(range(i * num_val_samples, (i + 1) * num_val_samples))
        val_fold = train_data.get_subset_for_days(val_days)

        train_days = (
            list(range(i * num_val_samples)) +
            list(range((i + 1) * num_val_samples, num_val_samples * k))
        )
        train_fold = train_data.get_subset_for_days(train_days)
        for clients_per_round in clients_per_round_list:
            model = model_class(
                parameter_config=parameter_dict,
                parameter_overwrite={"clients_per_round": clients_per_round}
            )
            results = run_single_experiment(
                model, train_fold, val_fold, plotAUC=False
            )
            aucs_by_clients[clients_per_round].append(results["AUC"])

    for clients_per_round in clients_per_round_list:
        aucs_by_clients[str(clients_per_round) + "_avg"] = np.mean(aucs_by_clients[clients_per_round])

    return aucs_by_clients

def main():

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

    clients_per_round_list = [50, 60, 65, 70, 75, 80]

    train_data, test_data = simple_train_test_split(parameter_dict)

    aucs_by_clients = run_cv_FedModel(model_class, train_data, k, clients_per_round_list, parameter_dict)

    write_to_json(aucs_by_clients, parameter_dict["output_path"] + "_cv_clients")


if __name__ == '__main__':
    main()

