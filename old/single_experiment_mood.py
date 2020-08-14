import datetime
import json
import sys

# import user-defined functions

from ExperimentUtils import ExperimentUtils


def main():
    start = datetime.datetime.now()
    
    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)


    train_data, test_data = ExperimentUtils.mood_train_test_split(
        parameter_dict
    )

    # get the name of the model from the user-inputted json file
    # and match it to the corresponding model object
    model_class = ExperimentUtils.model_from_config(
        parameter_dict['model_type']
    )
    model = model_class(parameter_config=parameter_dict)

    results = ExperimentUtils.run_single_experiment(
        model, train_data, test_data, parameter_dict['test_callback']
    )

    if parameter_dict['model_type'] != 'baseline' and parameter_dict['model_type'] != 'moving_mean_model':
        results['lr'] = parameter_dict['learn_rate']
    
    ind_results = model.individual_evaluate(test_data)

    if parameter_dict["plot_auc"]:

        ExperimentUtils.plot_auc(
            results["FPR"],
            results["TPR"],
            results["AUC"],
            str(
                parameter_dict["auc_output_path"] +
                "_(" + parameter_dict['model_type'] + ")"
            )
        )
    try:
        del results["FPR"]
        del results["TPR"]
    except KeyError:
        pass
    ExperimentUtils.write_to_json(
        ind_results,
        str(
            parameter_dict["output_path"] + "_by_user" +
            "_(" + parameter_dict['model_type'] + ")"
        )
    )
    ExperimentUtils.write_to_csv(
        results,
        str(
            parameter_dict["output_path"] +
            "_(" + parameter_dict['model_type'] + ")"
        )
    )
    ExperimentUtils.write_to_json(
        results,
        str(
            parameter_dict["output_path"] +
            "_(" + parameter_dict['model_type'] + ")"
        )
    )

    finish = datetime.datetime.now() - start
    print('Time to finish: ' + str(finish.total_seconds()))


if __name__ == '__main__':
    main()
