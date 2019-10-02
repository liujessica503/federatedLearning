'''Plot AUC after running globalmodel.py
Maintenance:
10/1/19 Created
'''

import matplotlib.pyplot as plt 

def plot_auc(auc_value):
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_value))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()