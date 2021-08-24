from numpy import load, zeros, linspace, save, vstack
import scipy.linalg as spl
from sys import argv, exit
from timeit import default_timer
import cmb_likelihood_utils as utils

if len(argv) < 2:
    print("Wrong number of input arguments.")
    print("Usage: python cmb_likelihood.py params.py")
    exit()

# Reading parameters from param file into namespace.
# Tip: this is one possible way of using a parameter file in python, allowing you to keep all your run-specific parameters in a separate file, 
# making it easier to keep track of what values you are using and de-cluttering your code.
# Disclaimer: This is not an optimal way, just _a_ way. It's unsafe in the sense that there's nothing stopping you from overwriting your parameters within the code, 
# which is bad practice (if they're to be considered constants, at least). Feel free to improve it! 

namespace = {}
paramfile = argv[1]
exec(open(paramfile).read())
globals().update(namespace)

runtime_start = default_timer()

print("Loading cmb data from input file %s"%cmbfile)
data = load(cmbfile)
x, y, z, cmb, rms = [data[:, i] for i in range(5)]

numdata = len(x)
print("Number of unmasked pixels to be used for analysis:", numdata)

print("Loading beam from file %s, using ells of 0 through %d"%(beamfile, lmax))
data = load(beamfile)
ells, beam = [data[:, i] for i in range(2)]
beam = beam[0 : lmax + 1]

print("Loading temperature pixel window from file %s, using ells of 0 through %d"%(pixwinfile, lmax))
data = load(pixwinfile)
ells, pixwin = [data[:, i] for i in range(2)]
pixwin = pixwin[0 : lmax + 1]

# Finished setup of input data
# --------------------------------------
print("Finished loading data. Now pre-computing noise and foreground covariances")
N_cov = utils.get_noise_cov(rms)
F_cov = utils.get_foreground_cov(x, y, z)

print("Now pre-computing Legendre polynomials for signal covariance")
p_ell_ij = utils.get_legendre_mat(lmax, x, y, z)

time_a = default_timer()
print("Time spent on setup: %f seconds"%(time_a - runtime_start))

# Finished precomputation
# ---------------------------------------
print("Starting likelihood evaluation loop")

# Defining grid based on param file values
lnL = zeros((q_numpoint, n_numpoint))
Q_values = linspace(q_min, q_max, q_numpoint)
n_values = linspace(n_min, n_max, n_numpoint)

# Making an output filename for ASCII format
resultfile_dat = resultfile[:resultfile.rfind(".")] + ".dat"

# Main computation loop
for i, Q in enumerate(Q_values):
    for j, n in enumerate(n_values):
        print("Now computing lnL for Q=%f, n=%f"%(Q, n))
        time_b = default_timer()
        # Computing model curve
        Cl_model = utils.get_C_ell_model(Q, n, lmax)
        time_c = default_timer()
        # Computing signal covariance matrix for current model
        S_cov = utils.get_signal_cov(Cl_model, beam, pixwin, p_ell_ij)
        # Assembling to total covariance matrix
        cov = S_cov + N_cov + F_cov
        time_d = default_timer()
        # Solving for loglikelihood
        lnL[i, j] = utils.get_lnL(cmb, cov)
        time_e = default_timer()

        if debug_mode:
            # Printing some time usage
            print("Time spent:")
            print("Model computation: %f sec"%(time_c - time_b))
            print("Signal cov computation: %f sec"%(time_d - time_c))
            print("Loglikelihood computation: %f sec"%(time_e - time_d))
            print("In total per grid point: %f sec"%(time_e - time_b))
            # In-loop printing of lnL, so we have something to look at
            # even if the job isn"t finished
            with open(resultfile_dat, "a") as outfile:
                outfile.write("%f %f %f\n"%(Q, n, - 0.5 * lnL[i, j]))


lnL *= - 0.5
print("Total runtime: %f seconds"%(default_timer() - runtime_start))

# Saving full likelihood in numpy array format. This is faster and easier
# to read in later, for visualization
save(resultfile, vstack([Q_values.T, n_values.T, lnL]))

# Adding a dump to ASCII file as well, in case you prefer non-python visualization
if not debug_mode:
    with open(resultfile_dat, "w") as outfile:
        for i, Q in enumerate(Q_values):
            for j, n in enumerate(n_values):
                outfile.write("%f %f %f\n"%(Q, n, lnL[i, j]))