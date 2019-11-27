import datetime
import json
import sys

# import user-defined functions

from ExperimentUtils import ExperimentUtils


def main():
    start = datetime.datetime.now()

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    train_data, test_data = ExperimentUtils.simple_train_test_split(
        parameter_dict
    )

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_class = ExperimentUtils.model_from_config(
        parameter_dict['model_type']
    )
    model = model_class(parameter_config=parameter_dict)

    results = ExperimentUtils.run_single_experiment(
        model, train_data, test_data
    )
    ind_results = model.individual_evaluate(test_data)

    if parameter_dict["plot_auc"]:

        ExperimentUtils.plot_auc(
            results["FPR"],
            results["TPR"],
            results["AUC"],
            str(
                parameter_dict["auc_output_path"] +
                "(" + parameter_dict['model_type'] + ")"
            )
        )
    del results["FPR"]
    del results["TPR"]
    ExperimentUtils.write_to_json(
        ind_results, parameter_dict["output_path"] + "_by_user"
    )
    ExperimentUtils.write_to_csv(
        results, parameter_dict["output_path"]
    )
    ExperimentUtils.write_to_json(
        results, parameter_dict["output_path"]
    )

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))


if __name__ == '__main__':
    main()
