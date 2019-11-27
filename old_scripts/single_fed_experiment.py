import datetime
import json
import sys

# import user-defined functions
from FedModel import FedModel

from single_global_experiment import (
    run_single_experiment,
    write_to_csv,
    write_to_json,
    simple_train_test_split,
)


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    train_data, test_data = simple_train_test_split(parameter_dict)

    model = FedModel(parameter_config=parameter_dict)

    results = run_single_experiment(model, train_data, test_data, plotAUC=True)

    write_to_csv(results, parameter_dict["output_path"])
    write_to_json(results, parameter_dict["output_path"])

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))


if __name__ == '__main__':
    main()
