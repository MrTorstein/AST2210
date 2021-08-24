from numpy import load, zeros, where, array, linspace
from matplotlib.pyplot import plot, show
from sys import argv, exit

if len(argv) < 3:
	print("To few arguments given")
	print("Called by giving numpy data file as second argument,")
	print("the number of your variabel to integrate over")
	print("and the number of variables")
	print("F.eks: py integrate_var.py cobe_dmr_53ghz_lnL.npy 0 2")
	exit()

inputfile 		= argv[1]
varnr			= int(argv[2])
nrvars			= int(argv[3])
a               = load(inputfile)
var				= zeros((len(a), len(a[0, :])))
lnL				= a[nrvars:]

plot(linspace(1, 50, 40), lnL)
show()

for i in range(len(a) - 1):
	var[i] = a[i]

v = zeros((len(var[0]), len(var[1])))
for i in range(len(var[varnr]) - 1):
	a = lnL[i]
	v[i + 1] = v[i] +  a * ( var[varnr].max() - var[varnr].min() / len(var[varnr]) )

print(v[-1])

pos = int(array(where(v[- 1] == v[-1].max())))
print(float(var[varnr, pos]))