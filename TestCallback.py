# after each epoch of training, 
# evaluate the model on our test set 

from typing import List, Any
import sys
import json
import keras.callbacks
#from BaseModel import BaseModel # for evaluate method
import csv # write test results to file

class TestCallback(keras.callbacks.Callback):

    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data

        with open(sys.argv[1]) as file:
        	parameter_dict = json.load(file)
        	loss_type = parameter_dict["output_layer"]["loss_type"]

        if loss_type == "regression":
        	loss, mae, mse = self.model.evaluate(x, y, verbose=0)
        	# write the test results to file
	        client_file_name = str(parameter_dict["output_path"] +
	            "_(" + parameter_dict['model_type'] + ")") + "test_per_epoch.csv"
	        with open(client_file_name, mode = 'a') as csvfile:
	            file_writer = csv.writer(csvfile, delimiter=',')
	            file_writer.writerow([loss, mae])
	        print('\nTesting loss: {}, mae: {}\n'.format(loss, mae))
        elif loss_type == "classification":

        	## NEED TO FIX THIS SO THAT MULTICLASS CAN RUN
        	## in order to do that, will need model: BaseModel, and use the basemodel evaluate function to get the evaluate: multiclass

        	loss, acc = self.model.evaluate(x, y, verbose=0)
        	 # write the test results to file
	        client_file_name = str(parameter_dict["output_path"] +
	            "_(" + parameter_dict['model_type'] + ")") + "test_per_epoch.csv"
	        with open(client_file_name, mode = 'a') as csvfile:
	            file_writer = csv.writer(csvfile, delimiter=',')
	            file_writer.writerow([loss, acc])
	        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

