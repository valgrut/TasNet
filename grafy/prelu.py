from matplotlib import pyplot
 
a = 0.1

# rectified linear function
def rectified(x):
	return max(a*x, x)
 
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
# line plot of raw inputs to rectified outputs
pyplot.grid()
pyplot.xlabel("x")
pyplot.ylabel("PReLU(x)") 
pyplot.plot(series_in, series_out)
pyplot.show()
