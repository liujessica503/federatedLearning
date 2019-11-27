import json
import csv
import matplotlib.pyplot as plt

# import user-defined functions
from split_train_test_global import split_train_test_global
from UserDayData import UserDayData
from typing import Any, Dict

from IndividualModel import IndividualModel
from GlobalModel import GlobalModel
from FedModel import FedModel
from BaseModel import BaseModel


class ExperimentUtils:

    model_registry = {
        "individual_model": IndividualModel,
        "global_model": GlobalModel,
        "fed_model": FedModel,
    }

    @staticmethod
    def run_single_experiment(
        model: BaseModel,
        train_data: UserDayData,
        test_data: UserDayData,
    )->Dict:
        model.train(train_data)
        metrics = model.evaluate(test_data)
        return metrics

    @staticmethod
    def write_to_csv(results, output_path: str)->None:
        csv_columns = list(results.keys())
        with open(output_path + ".csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(results)

        print('results written to :' + output_path + ".csv")

    @staticmethod
    def write_to_json(results, output_path: str)->None:
        with open(output_path + ".json", "w") as f:
            json.dump(results, f, indent=4)
        print('results written to :' + output_path + ".json")

    @staticmethod
    def simple_train_test_split(parameter_dict: Dict[str, Any])->UserDayData:
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
            X=train_covariates,
            y=train_labels,
            user_day_pairs=train_user_day_pairs
        )
        test_data = UserDayData(
            X=test_covariates,
            y=test_labels,
            user_day_pairs=test_user_day_pairs
        )

        return train_data, test_data

    @staticmethod
    def model_from_config(model_type):
        try:
            model_class = ExperimentUtils.model_registry[model_type]
            return model_class
        except KeyError:
            raise KeyError(
                'model_type in config json must be one of: "individual_model",'
                '"global_model", "fed_model"'
            )

    @staticmethod
    def plot_auc(fpr, tpr, auc_value, filename):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_value))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig(filename + ".pdf")
