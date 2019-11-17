'''
To-do: in split_train_test, need to add a function for split_train_valid_test
Need to check if we have enough test / valid data before splitting.
'''

# later add graphics from analyzing results

import json
import sys
import csv

# import user-defined functions
from split_train_test_global import split_train_test_global
from GlobalModel import GlobalModel
from UserDayData import UserDayData
from typing import Any, Dict
from BaseModel import BaseModel

import datetime


def run_single_experiment(
    model: BaseModel,
    train_data: UserDayData,
    test_data: UserDayData,
    plotAUC=False
)->Dict:
    model.train(train_data)
    metrics = model.evaluate(test_data, plotAUC=plotAUC)
    return metrics


def write_to_csv(results, output_path: str)->None:
    csv_columns = list(results.keys())
    with open(output_path + ".csv", 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(results)

    print('results written to :' + output_path + ".csv")


def write_to_json(results, output_path: str)->None:
    with open(output_path + ".json", "w") as f:
        json.dump(results, f, indent=4)
    print('results written to :' + output_path + ".json")


def simple_train_test_split(parameter_dict: Dict[Any])->UserDayData:
    (
        train_covariates,
        train_labels,
        train_user_day_pairs,
        test_covariates,
        test_labels,
        test_user_day_pairs
    ) = split_train_test_global(
        directory=parameter_dict['input_directory'],
        cv=parameter_dict['cv']
    )

    train_data = UserDayData(
        X=train_covariates, y=train_labels, user_day_pairs=train_user_day_pairs
    )
    test_data = UserDayData(
        X=test_covariates, y=test_labels, user_day_pairs=test_user_day_pairs
    )

    return train_data, test_data


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    train_data, test_data = simple_train_test_split(parameter_dict)

    model = GlobalModel(parameter_config=parameter_dict)

    results = run_single_experiment(model, train_data, test_data, plotAUC=True)

    write_to_csv(results, parameter_dict["output_path"])
    write_to_json(results, parameter_dict["output_path"])

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))


if __name__ == '__main__':
    main()
