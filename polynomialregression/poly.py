import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

def coefs():
	'''
	returns a numpy array of 5 random numbers chosen uniformly from the range -.5,.5
	'''
	return np.array(np.random.uniform(-.5, .5, 5))


def makeTestData(coef):
	'''
	takes coeffecients for a 4th degree polynomial (5 coeffecients)
	and returns (X,y) where 
	x is 100 points evenly spaced from -2, 2
	y is the output for the corresponding polynomial
	'''
	X = np.linspace(-2, 2, 100)
	return (X, np.array([np.sum(coef * np.array([x ** i for i in range(5)])) for x in X]) )


def makeTrainingData(n, coef):
	'''
	creates the traning data for out linear regression
	randomness created by adding noise with mean 0 standard deviation 1 to each y value

	'''
	X = 4 * np.random.rand(n) - 2
	return (X, (np.array([np.sum(coef * np.array([x ** i for i in range(5)])) for x in X])) + np.random.randn(n))


def fitPoly(X, y, deg):
	'''
	takes testing data X, y, and the degree of the polynomial regression,
	returns plottable X, y from the regression's predictions

	'''
	X = np.reshape(X, (-1, 1))
	poly_features = PolynomialFeatures(degree=deg, include_bias=True)
	X_poly = poly_features.fit_transform(X)
	lin_reg = lm.LinearRegression()
	lin_reg.fit(X_poly, y)
	X1 = np.reshape(np.linspace(-2, 2, 1000), (-1, 1))
	X2 = poly_features.fit_transform(X1)
	return (X1, lin_reg.predict(X2))


def MSE(X, y, X1, y1, deg):
	'''
	takes training data (X,y), testing data (X1, y1) and degree of regression (deg)
	returns a tuple of mean squared error for the linear regression on the training set and testing set
	'''
	X = np.reshape(X, (-1, 1))
	poly_features = PolynomialFeatures(degree=deg, include_bias=True)
	X_poly = poly_features.fit_transform(X)
	lin_reg = lm.LinearRegression()
	lin_reg.fit(X_poly, y)
	X1 = np.reshape(X1, (-1, 1))
	X1_poly = poly_features.fit_transform(X1)
	yp = lin_reg.predict(X_poly)
	y1p = lin_reg.predict(X1_poly)
	return (mean_squared_error(y, yp), mean_squared_error(y1, y1p))


def regressionTest(coef = coefs(), degrees = (1,2,20), n_points = (10,100,1000), colors = ('red','blue','green')):
	'''
	takes a list of coeficients, a list of degrees, a list of data point amounts
	and a list of colors coresponding to the graphs of each degree
	
	for each n points in n_points, plots a graph for each degree in degrees
	'''
	test_X, test_y = makeTestData(coef)
	titleList = ['Regressions of '+ str(n) + ' Data Points' for n in n_points]
	for i,n in enumerate(n_points):
		X, y = makeTrainingData(n, coef)
		plt.scatter(X, y, s=(3 / (i + 1)), color="black")
		plt.plot(test_X, test_y, color="purple", label="Test curve")
		plt.title(titleList[i])
		plt.ylim(-10, 10)
		plt.xlabel('X')
		plt.ylabel('y')
		for k,d in enumerate(degrees):
			X1, y1 = fitPoly(X,y,d)
			plt.plot(X1, y1, color=colors[k], label='Degree '+str(d), linewidth=(i + 1) ** .5)

		plt.legend()
		plt.show()

def mseTest(coef_func = coefs, n_points = (10, 100, 1000), maxDegree = 20):
	'''
	takes a function for generating coeffecients, a list of data point amounts
	and a list of degrees

	for each n points in n_points, 
	plots the average mean squared error (over 100 tests) for testing and traning data (generated by coef_func)
	for each degree up to maxDegree
	'''
	MSEs = np.zeros((3, 21, 3))
	for i in range(100):
		coef = coef_func()
		for i, n in enumerate(n_points):
			X1, y1 = makeTestData(coef)
			X, y = makeTrainingData(n, coef)
			MSEs[i] = np.add(MSEs[i], np.array([((d,) + MSE(X, y, X1, y1, d)) for d in range(maxDegree+1)]))

	MSEs /= 100
	for i in range(3):
		plt.plot(MSEs[i][:, 0], MSEs[i][:, 1], label="Training MSE")
		plt.plot(MSEs[i][:, 0], MSEs[i][:, 2], label="Testing MSE")
		plt.title("Mean Squared Error for " + str(10 ** (i+1)) + " Points")
		plt.xlabel('Degree')
		plt.ylabel('Average Mean Squared Error \n (Over 100 Tests)')
		plt.legend()
		plt.xticks((range(21)))
		plt.ylim(0, 10)
		plt.show()

if __name__ == "__main__":
	regressionTest()
	mseTest()
