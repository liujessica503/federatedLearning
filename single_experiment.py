import datetime
import json
import sys

# import user-defined functions
from IndividualModel import IndividualModel
from GlobalModel import GlobalModel
from FedModel import FedModel

from run_single_experiment import (
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

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_registry = {
        "individual_model": IndividualModel,
        "global_model": GlobalModel,
        "fed_model": FedModel,
    }

    if model_registry.get(parameter_dict.get('model_type')) is None:
        raise KeyError(
            'model_type in .json must be one of: "individual_model",'
            '"global_model", "fed_model"'
        )
    else:
        model_class = model_registry[parameter_dict['model_type']]

    if model_class == IndividualModel:
        model = IndividualModel(parameter_config=parameter_dict)
        results = run_single_experiment(
            model, train_data, test_data, plotAUC=True
        )
        ind_results = model.individual_evaluate(test_data)
        write_to_json(ind_results, parameter_dict["output_path"] + "ind")

    elif model_class == FedModel:
        model = FedModel(parameter_config=parameter_dict)
        results = run_single_experiment(
            model, train_data, test_data, plotAUC=True
        )

    elif model_class == GlobalModel:
        model = GlobalModel(parameter_config=parameter_dict)

    write_to_csv(results, parameter_dict["output_path"])
    write_to_json(results, parameter_dict["output_path"])

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))


if __name__ == '__main__':
    main()
