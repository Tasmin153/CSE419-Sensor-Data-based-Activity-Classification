import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import figure, axes, pie, title, show


def input_data_plot(X_train,X_test,xx,y_test,y_train,yy):

	figure = plt.figure()
		    
	# just plot the dataset first
	cm = plt.cm.RdBu
	cm_bright = ListedColormap(['#FF0000', '#0000FF'])
		    #ax = plt.plot(len(file_list), len(classifiers) + 1, i)
	ax = figure.add_axes([0.1, 0.1, 0.8, 0.8])
	ax.set_title("Input data")
		    # Plot the training points
	ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='k')
		    # Plot the testing points
	ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,edgecolors='k')
	ax.set_xlim(xx.min(), xx.max())
	ax.set_ylim(yy.min(), yy.max())
	ax.set_xticks(())
	ax.set_yticks(())
	plt.savefig('train-test.png')

  
    