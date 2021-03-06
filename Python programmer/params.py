# PARAMETER FILE FOR CMB_LIKELIHOOD

# Basic run parameters
debug_mode	= True # or False. debug_mode prints lnL to .dat file in-loop, and prints time usage
#debug_mode	= False
nside		= 16
lmax		= 47
band		= 53 # or 90, in GHz
#band    	= 90

# Grid specifications
q_min      =   1.
q_max      =  50.
q_numpoint =  40
n_min      =  - 1.
n_max      =   3.
n_numpoint =  40

# Input and output file paths
cmbfile    = "data/cobe_dmr_%dGHz_n%d.npy"%(band, nside)
beamfile   = "data/cobe_dmr_beam.npy"
pixwinfile = "data/pixwin_n%d.npy"%nside

resultfile = "cobe_dmr_%dghz_lnL_2.npy"%band