import os
import scipy as sp
import matplotlib.pyplot as plt


data_dir = os.path.join( os.path.dirname(os.path.realpath(__file__)),"..","data")

data = sp.genfromtxt(os.path.join( data.dir, "web_traffic.tsv"), delimiter = "\t")
print(data[:10])


colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']
x = data[:,1]
y = data[:,2]


print("Number of invalid entries", sp.sum(isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#plot input data

def plot_models(x, y, models, fname ,mx = None , ymax = None, xmin = None):

	plt.clf()
	plt.scatter(x,y)
	plt.title("Web traffic over the last month")
	plt.xlabel("Time")
	plt.ylabel("Hits/hour")
	plt.xticks([w*7*24 for w in xrange(10)],['week %i' % w for w in xrange(10)]) 
	

	if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            # print "Model:",model
            # print "Coeffs:",model.coeffs
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)
	
		plt.legend(["d = %i" %m.order for m in models], loc = "upper left")
	plt.autoscale( tight = True )
	plt.ylim( ymin  = 0)

	if ymax:
		plt.ylim(ymax = ymax )
	if xmin:
		plt.xlim( xmin  = xmin)
	plt.grid( True, linestyle = '-', color='0.75')
	plt.savefig(fname)

#first look at the data
plot_models(x , y, None, os.path.join("..","1400_01_01.png"))

#create and plot models
fp1, res, rank, sc, rcond = sp.polyfit(x,y,1, full=Mrue)
print("Model Parameters : %s" %fp1)
print("Error of the model:", res)
f1 = sp.poly1d(fp1)
f2 = sp.poly1d(sp.polyfit(x,y,2))
f3 = sp.poly1d(sp.polyfit(x,y,3))
f10 = sp.poly1d(sp.polyfit(x,y,10))
f100 = sp.poly1d(sp.polyfit(x,y,100))

plot_models(x,y, [f1] , os.path.join("..","1400_01_02.png"))
plot_models(x,y, [f1,f2] , os.path.join("..","1400_01_03.png"))
plot_models(x,y, [f1, f2,f3,f10,f100] , os.path.join("..","1400_01_04.png"))

#fit and plot a model using the knowledge about inflection point
infletion = 3.5 * 7 * 24; #calculate the inflection point in hours
xa = x[: inflection]
ya = y[: inflection]
xb = x[inflection : ]
yb = y[inflection ; ]

fa = sp.poly1d(sp.polyfit(xa,ya,1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

plot_models(x,y,[fa,fb], os.path.join("..","1400_01_05.png"))


def error(f,x,y):
	return sp.sum((f(x)-y)**2)

print("Errors for the complete data set:")
for f  in [f1, f2, f3, f10, f100]:
	print("Error d=%i :  %f"%(f.order, error(f,x,y))

print("Errors for only the inflection point")
for f in [f1,f2,f3,f10,f100]:
	print("Error d = %i : %f" %(f.order, error(f,xb,yb)))

plot_models(
	x,y,[fb1, fb2, fb3, fb10, fb100], os.path.join("..","1400_01_07.png"),
	mx = sp.linspace(0, 6 * 7 *24, 100),
	ymax = 10000, xmin = 0
	)

#separating trainig from testing data

frac = 0.3

split_idx = int(frac * len(xb))
shuffled = sp.random.permutation(list(range(len(xb))))
test = sorted(shuffled[:split_idx])
train = sorted(suffled[splt_idx:])
fbt1 = sp.poly1d(sp.polyfit(xb[train], yb[train],1))
fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train],2))
fbt3 = sp.poly1d(sp.polyfit(xb[train], yb[train],3))
fbt10 = sp.poly1d(sp.polyfit(xb[train], yb[train],10))
fbt100 = sp.poly1d(sp.polyfit(xb[train], yb[train],100))

print("Test Errors for only the time after inflection point")
for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
	print("Error d=%i : %f" %(f.order, error(f, xb[test], yb[test])))

plot_models(
	x,y, [fbt1, fbt2, fbt3, fbt10, fbt100], os.path.join("..","1400_01_08.png"),
	mx = sp.linspace(0,6*7*24,100),
	ymax = 10000, xmin = 0
	)
from scipy.optimize import fsolve
print(fbt2)
print(fbt2 - 100000 )
reached_max = fsolve(fbt2 - 10000,800)/(7*24)
print("100,100 hits/hour expected at week %f" % reached_max[0])