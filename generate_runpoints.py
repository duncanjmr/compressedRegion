import matplotlib.pyplot as plt
import os
from susy_tools import *
import multiprocessing as mp
from scipy.optimize import curve_fit
from PIL import Image
import matplotlib
from matplotlib.patches import Patch
from scipy.optimize import curve_fit
from scipy.interpolate import interp2d
import json



########################## Input Parameters ##########################
tanB = 50
m_sleptons = 700
N = (8,8)
min_diff = 2
max_diff = 70

min_M2 = 100
max_M2 = 250
sign_M1 = -1.

use_micromegas = True
gm2_sigma_target = 0
gm2_sigma_tol = 0.2

show_plot = True

######################### Initial Setup ##################################

print()
print("Generating Checkmate Run Points for:")
print("\t tanB: %i" % tanB)
print("\t m_sleptons: %i" % m_sleptons)
print("\t M2 Limits: (%i, %i)" % (min_M2, max_M2))
print("\t Sign of M1: %i" % (sign_M1))
print()

## Edit susy-hit input to generate spectra
changeParamValue("A_t", 3.50e3)
changeParamValue("A_tau", -250)
changeParamValue("tanbeta(MZ)", tanB)

changeParamValue("M_eL",   m_sleptons)
changeParamValue("M_eR",   m_sleptons)
changeParamValue("M_muL",  m_sleptons)
changeParamValue("M_muR",  m_sleptons)
changeParamValue("M_tauL", 2.5e3)
changeParamValue("M_tauR", 2.5e3)

direc = "%i_%i_%i" % (tanB, m_sleptons, sign_M1)
if direc not in os.listdir("./"):
    os.mkdir(direc)
    
M2_init = np.linspace(min_M2, max_M2, N[0])

print("Results will be saved in ./%s" % direc)
print()
    
################ Scan Point Generation #######################

print("Running initial mu estimation...")
mu_init = optimize_gm2_array([[sign_M1*M2, M2] for M2 in M2_init], m_sleptons, 
                             tanB, verbose=False, use_micromegas=use_micromegas, 
                             sigma_target=gm2_sigma_target, sigma_tolerance=0.05, cap=4e3)

print("Done.")
print()

print("Calculating minima of delta(x1,x2) for each M2: \n")
print(" M2 \t  mu \t M1_min\t delta m(x1,x2)")
print("------------------------")

M1_minimized = []
M2_l = []
mu = []
minMassSplitting = []
sh_corr = []
 
for i, M2 in enumerate(M2_init):
    
    print("%i: \t %i" % (M2, mu_init[i]), end="")
    changeParamValue("mu(EWSB)", mu_init[i])

    # Now find the mass splitting minimum.
    M_min, diff, corr = minimize_neutralinoMassDiff(sign_M1*M2, M2, return_diff=True,verbose=False)
    
    if np.isnan(M_min[0]):
        print("\t nan")
        continue
        
    M2_l.append(M2)
    mu.append(mu_init[i])
    M1_minimized.append(M_min[0])
    minMassSplitting.append(diff)
    sh_corr.append(corr)

    
    if diff > 0.5:
        print("\t %i \t %.1f" % (M_min[0],  diff))
    else:
        print("\t %i \t %.1E" % (M_min[0],  diff))


points_l = []
# Generate columns of points for each m2
for i in range(len(M1_minimized)):

    # Get range of column of points in M1 space
    m1_max = M1_minimized[i] - sign_M1 * (minMassSplitting[i] < min_diff) * (min_diff - minMassSplitting[i])
    m1_min = m1_max - sign_M1 * max_diff

    # Adjust number of points to generate if mass splitting has a large minimum
    N_adj = int(np.floor(N[1] * np.log(max_diff / max(minMassSplitting[i], min_diff)) / np.log(max_diff / min_diff)))
    
    # Generate column
    dm_points = np.logspace(*np.log10([min_diff, max_diff]), N_adj)
    m1_pts = m1_max - sign_M1 * (dm_points - min_diff)
        
    points_l.append(np.vstack((m1_pts, M2_l[i] * np.ones(N_adj))).T)


mu_l = optimize_gm2_array(np.vstack(points_l), m_sleptons, tanB, 
                          sigma_target=gm2_sigma_target, sigma_tolerance=0.2, 
                          verbose=True, cap=1e3, use_micromegas=use_micromegas)

def additional_command(M1, M2, index):
    changeParamValue("mu(EWSB)", mu_l[index])

# Run susyhit and micromegas
data = run(np.vstack(points_l), True, run_prospino=False, 
               run_micromegas=True,
               run_checkmate=False,
               working_directory="./%s" % (direc),
               additional_command=additional_command)


br_gam = []
points_new = []

# Extract the photon branching ratio for plot showing 
for i in range(len(data["M2"])):

    filename = "spectrum_%i_%i.dat" % (data["M1"][i], data["M2"][i])
    
    with open("./%s/spectra_slha/" % direc + filename) as f:
        s = f.read()

    points_new.append([data["M1"][i], data["M2"][i]])
    br_gam.append(np.nan_to_num(getBranchingRatio(s, "BR(~chi_20 -> ~chi_10 gam)")))

######################################################
################ Plot points and show ################
######################################################

x = np.array(data["M2"])
y = np.abs(data["~chi_20"]) - np.abs(data["~chi_10"])
sel = ~np.isnan(y) * ~np.isnan(x)

gm2 = data["gm2"]
om = data["omega_dm"]
dd = np.ones(len(gm2))

f = plt.figure(figsize=(10, 5))
ax = f.add_subplot(1, 2, 1)

show_omega=True

name = "$\chi_1^0 + \gamma$"
levels = np.linspace(0,1, 11)

br = np.array(br_gam)
select = ~np.isnan(br) * sel

xsel = x[select]
ysel = y[select]
br_nonan = br[select]

mx = max(br_nonan)
if np.all(br_nonan == 0):
    mn = 0
else:
    mn = 0.01*min(br_nonan[br_nonan != 0])

cmap_br = truncate_colormap(plt.get_cmap("Reds"), mn, np.min([mx, 1.]))

cs = ax.tricontourf(xsel, ysel, br_nonan, levels=levels, 
                alpha=0.5, cmap=cmap_br)

# Plot contour lines
cmapblack = truncate_colormap(plt.get_cmap("Greys"), 0.9, 1)
cs = ax.tricontour(xsel, ysel, br_nonan, levels=levels,
                  cmap=cmapblack, linewidths=1)

ax.clabel(cs, cs.levels, inline=True, fontsize=10)

ax.scatter(xsel, ysel)


# Plot omega_dm bounds
s2 = ~np.isnan(om)[sel]
ax.tricontourf(x[s2], y[s2], np.array(om)[sel][s2], levels=[0.07, 0.3], cmap="Purples", alpha=0.8)

# Plot direct detection exclusion
plt.tricontourf(x, y, np.array(dd)[sel], levels = [0, 0.05], cmap="Greys",alpha=0.7)

# Set some plot labels
ax.set_xlabel(r"M2 ($\mu$ fixed to g-2) [GeV]", fontsize=12)
ax.set_title(r"BR$(\chi_2^0\  \to $ %s)" % name, fontsize=16)
ax.set_yscale("log")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(1)

ax.set_ylabel(r"$\Delta(\chi_1^0, \chi_2^0)$ [GeV]", fontsize=12)


# Plot mu vs M2 
ax = f.add_subplot(1, 2, 2)
ax.plot(M2_l, mu)
ax.set_xscale("log")
ax.set_title("M2 vs. mu", fontsize=15)
ax.set_xlabel("M2 [GeV]", fontsize=12)
ax.set_ylabel("mu [GeV]", fontsize=12)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, which="both")

sign = "\nM1<0"
plt.text(np.max(x)*0.75, ax.get_ylim()[1]*0.9, 
         "tanB: %i\nm_sl: %i GeV"% (tanB, m_sleptons) + sign*int(sign_M1 < 0), 
         fontsize=12, bbox=dict(facecolor='white', edgecolor='black',))
# #ax.legend([Patch(facecolor='purple', label='Color Patch', alpha=0.4),
#            Patch(facecolor='grey', label='Color Patch', alpha=0.4),
#            cm_plot
#           ],
#           [r"$\Omega = 0.07 - 0.3$",
#            r"DD p-value < 0.05",
#            r"Checkmate Run Points",], fontsize=12, loc=3)


plt.tight_layout()


# Save, open image, and then remove it
if show_plot:
    plt.savefig("%s/generated_points.png" % direc)

    img = Image.open('%s/generated_points.png' % direc)
    img.show() 
