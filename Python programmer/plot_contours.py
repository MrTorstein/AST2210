# Simple contour plotting script for visualizing the lnL computed by
# cmb_likelihood.py. 
# For convenience, it takes as input either the .npy file or the .dat file.
# In the .dat case you also have to supply the number of grid points in each 
# direction so that we can define the grid correctly.

from numpy import load, meshgrid, loadtxt, reshape, amax, where
from matplotlib.pyplot import contour, grid, title, xlabel, ylabel, show, savefig, figure, plot, legend, clabel
from sys import argv, exit

if len(argv) < 2:
    print('Wrong number if input arguments.')
    print('Usage: python plot_contours.py resultfile.npy')
    print('Or: python plot_contours.py resultfile.dat numpoints_Q numpoints_n')
    exit()

figure()
print("n                Q         lnL")
for i in range(1, len(argv)):
    inputfile = argv[i]
    
    if inputfile[inputfile.rfind('.'):] == '.npy':  #numpy file
        a               = load(inputfile)
        Q_values        = a[0, :]
        n_values        = a[1, :]
        lnL             = a[2:, :]
        qgrid, ngrid    = meshgrid(Q_values, n_values, indexing = 'ij')

    else: # ascii file
        n_Q         = int(argv[2])
        n_n         = int(argv[3])
        a           = loadtxt(inputfile)
        qgrid       = reshape(a[:, 0],(n_Q, n_n))
        ngrid       = reshape(a[:, 1],(n_Q, n_n))
        lnL         = reshape(a[:, 2],(n_Q, n_n))
        Q_values    = qgrid[:, 0]
        n_values    = ngrid[0, :]

    lnL -= amax(lnL) # arbitrarily "normalizing" to make the numbers more manageable

    # For a Gaussian distribution, the 1, 2 and 3 sigma (68%, 95% and
    # 99.7%) confidence regions correspond to where -2 lnL increases by
    # 2.3, 6.17 and 11.8 from its minimum value. 0.1 is close to the peak. 
    my_levels = [0.1, 2.3, 6.17, 11.8]
    cs = contour(ngrid, qgrid, -2. * lnL, levels = my_levels, colors = 'k')

grid()
title("Contour plot of liklihood of the of parameters Q and n")
xlabel("n")
ylabel("Q")
#savefig("Figur01.png")

show()
