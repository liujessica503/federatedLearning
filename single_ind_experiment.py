import datetime
import json
import sys

# import user-defined functions
from split_train_test_global import split_train_test_global
from IndividualModel import IndividualModel
from UserDayData import UserDayData

from single_global_experiment import (
    run_single_experiment,
    write_to_csv,
    write_to_json,
)


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

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

    model = IndividualModel(parameter_config=parameter_dict)

    results = run_single_experiment(model, train_data, test_data, plotAUC=True)
    ind_results = model.individual_evaluate(test_data)

    write_to_csv(results, parameter_dict["output_path"])
    write_to_json(results, parameter_dict["output_path"])

    write_to_json(ind_results, parameter_dict["output_path"] + "ind")

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))


if __name__ == '__main__':
    main()
