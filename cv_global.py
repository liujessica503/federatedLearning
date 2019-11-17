import json
import sys
import numpy as np

# import user-defined functions
from GlobalModel import GlobalModel
from collections import defaultdict
from single_global_experiment import (
    run_single_experiment,
    write_to_json,
    simple_train_test_split,
)


def main():

    with open(sys.argv[1]) as file:
        parameter_dict = json.load(file)

    k = 3
    # split our 142 days of training data into k partitions
    num_val_samples = (71 * parameter_dict["cv"]) // k

    lrs = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
    aucs_by_lr = defaultdict(list)

    train_data, test_data = simple_train_test_split(parameter_dict)

    for i in range(k):
        val_days = list(range(i * num_val_samples, (i + 1) * num_val_samples))
        val_fold = train_data.get_subset_for_days(val_days)

        train_days = (
            list(range(i * num_val_samples)) +
            list(range((i + 1) * num_val_samples, num_val_samples * k))
        )
        train_fold = train_data.get_subset_for_days(train_days)
        for lr in lrs:
            model = GlobalModel(
                parameter_config=parameter_dict,
                parameter_overwrite={"lr": lr}
            )
            results = run_single_experiment(
                model, train_fold, val_fold, plotAUC=False
            )
            aucs_by_lr[lr].append(results["AUC"])

    for lr in lrs:
        aucs_by_lr[str(lr) + "_avg"] = np.mean(aucs_by_lr[lr])

    write_to_json(aucs_by_lr, parameter_dict["output_path"] + "_cv")


if __name__ == '__main__':
    main()
