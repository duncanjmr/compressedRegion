import numpy as np
import matplotlib.colors as colors
import os
import multiprocessing as mp
import queue
import time
import datetime
import warnings
from scipy.interpolate import interp2d
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import nearest_points
from scipy.optimize import minimize



import subprocess
from subprocess import DEVNULL, STDOUT, check_call


scripts_dir = "/home/duncan/UChicago/SUSY/home"
checkmate_dir = "/home/duncan/UChicago/Software/checkmate2"
susyhit_dir = "/home/duncan/UChicago/SUSY/susyhit"
micromegas_dir = "/home/duncan/UChicago/SUSY/micromegas"
suspect_dir = "/home/duncan/UChicago/SUSY/suspect3/"


def changeParamValue(label, value):
    """
    Changes a parameter value in suspect2_lha.in.
    
    label is the comment attached to the line which is to be edited in the susyhit input file.
    """
    
    with open(susyhit_dir + "/suspect2_lha.in", "r") as f:
        s = f.read()

    end = s.find(' # ' + label)
    
    if end == -1:
        print("Finding parameter %s failed." % label)
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        parind = len(line) - len(line.lstrip())
        ind_val = len(line) - len(line[parind+3:].lstrip())
        line = line[:ind_val] + "{:.6e}".format(value)
        s = s[:start] + line + s[end:]
        with open(susyhit_dir + "/suspect2_lha.in", "w") as f:
            f.write(s)

def changeParamValueStr(s, label, value):
    """
    Changes a parameter value in suspect2_lha.in.
    
    label is the comment attached to the line which is to be edited in the susyhit input file.
    """

    end = s.find(' # ' + label)
    
    if end == -1:
        print("Finding parameter %s failed." % label)
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        parind = len(line) - len(line.lstrip())
        ind_val = len(line) - len(line[parind+3:].lstrip())
        line = line[:ind_val] + "{:.6e}".format(value)
        s = s[:start] + line + s[end:]
        
    return s
            
def getParamValue(s, label):

    """
    Gets a parameter value, labeled by the comment (hashtag not included), in the susyhit_slha.out file.
    Note that one has to open the file and read it as a string before feeding it to this function.
    """
    end = s.find(' # ' + label)
    
    if end == -1:
        #warnings.warn("Finding parameter %s failed. Returning NaN." % label)
        return np.nan
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        
        return float(line.split()[-1])
    
def getBranchingRatio(s, label):
    """
    Similar to last function but instead of reading a parameter it reads a branching ratio.
    """
    
    end = s.find(' # ' + label)
    
    if end == -1:
        #warnings.warn("Finding parameter %s failed. Returning NaN." % label)
        return np.nan
        
    else:
        start = s[:end].rfind("\n") + 2

        line = s[start:end]

        if line[0] == "#": line[0] = " "

        parind = len(line) - len(line.lstrip())
        parlen = line[parind:].find(" ")
        
        ind_val = len(line) - len(line[parind+parlen:].lstrip())
        return float(line[parind: parind+parlen])
    
def getDecayWidth(s, ID):
    st = "DECAY   %s" % ID
    ind = s.find(st)
    if ind == -1:
        return np.nan
    
    end = s[ind:].find("#") + ind
    return float(s[ind + len(st): end])

def generate_parameter_points(M1_max, M2_max, n_M1=8, n_M2=8, randomize=False, M1_min=10, M2_offset=0):
    """ 
    Generates a grid of parameter points in the (M1, M2) plane.
    
    These points are constrained by M2 > M1 + M2_offset, and M1 > M1_min
    """
    M1_sign = np.sign(M1_max)
    M2_sign = np.sign(M2_max)
    
    M1_max = np.abs(M1_max)
    M2_max = np.abs(M2_max)
    
    M1_l = np.linspace(M1_min, M1_max, n_M1)
    
    dM1 = (M1_max - M1_min) / n_M1
    dM2 = (M2_max - M2_offset) / n_M2

    first = True
    for M1 in M1_l:
        
        mn = M1 + M2_offset
        
        N = int(np.ceil(n_M2 * (M2_max - mn)/ (M2_max-M2_offset)))
        M2_l = np.abs(np.logspace(*np.log10([mn, M2_max]), N) + 
                      randomize * np.random.uniform(-dM2/2, dM2/2, N))
        
        p = np.vstack((M1_sign * np.abs(np.ones(N) * M1 + 
                                        randomize * np.random.uniform(-dM1/2, dM1/2, N)), 
                       M2_sign * M2_l)).T    
        if first:
            points = p
            first = False
        else:
            points = np.vstack((points, p))
            
    return points

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Given a color map, returns a new colormap which has been sliced between the two values. 
    One can choose a subsection of a colormap for more pretty plots.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def get_higgs(A_t):
    """
    Runs susy-hit and reads the higgs mass, for a specific value of A_t, 
    the primary parameter for adjusting higgs mass.
    """
    changeParamValue("A_t", A_t)
    os.chdir("../new_susyhit")
    os.system("./run >> ../scripts/temp.out")
    
    # Get particle masses from susyhit output
    with open("susyhit_slha.out", "r") as f:
        s = f.read()
        h = getParamValue(s, "h")
    os.chdir("../scripts")
    return h

def optimize_higgs(h_target=125., tolerance=0.5):
    """
    Optimizes the higgs mass to a target value (usuall 125 GeV). 
    Useful if your parameters affect the higgs mass and you want to correct it.
    """
    
    x_0 = 3.0e3
    step = tolerance / h_target * x_0 / 2
    
    x = [x_0]
    h = [get_higgs(x_0)]

    c = 0
    while np.abs(h[-1] - h_target)  > tolerance and c < 5:
        hp = get_higgs(x[-1] + step)
        dh_dx = -(hp - h[-1]) / step
        
        if dh_dx == 0:
            break
        
        dx = (h[-1] - h_target) / dh_dx
        if abs(dx) > 1e3:
            dx = dx / abs(dx) * 1e3
        x.append(x[-1] + dx)
        h.append(get_higgs(x[-1]))
        c += 1

    return x[-1], h[-1]

def get_neutralino_eigenvalues(m1, m2, tanB, mu, ordered=True):

    sw = np.sqrt(0.222)
    cw = np.sqrt(1-sw**2)

    beta = np.arctan(tanB)
    MZ = 92.

    Massmatrix = np.array([[m1*cw**2 + m2 * sw**2, (m2-m1) *sw*cw, 0, 0],
                           [(m2-m1)*sw*cw, m1*sw**2 + m2*cw**2, MZ, 0],
                           [0, MZ, mu*np.sin(2*beta), -mu*np.cos(2*beta)],
                           [0,  0, -mu*np.cos(2*beta), -mu*np.sin(2*beta)]])


    masses = np.linalg.eig(Massmatrix)[0]
    
    if not ordered:
        return masses
    
    order = np.argsort(np.abs(masses))
    return masses[order]

def get_neutralinoMassDiff(M1, M2):
    """
    For given value of M1, M2, returns delta(x10, x20).
    """
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)

    os.chdir(susyhit_dir)
    os.system("./run >> " + scripts_dir + "/temp.out")
    
    # Get particle masses from susyhit output
    with open("susyhit_slha.out", "r") as f:
        s = f.read()
        x2 = getParamValue(s, "~chi_20")
        x1 = getParamValue(s, "~chi_10")
        
    os.chdir(scripts_dir)
    return np.abs(x2) - np.abs(x1)

def get_neutralinoMasses(M1, M2):
    """
    For given value of M1, M2, returns delta(x10, x20).
    """
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)

    os.chdir(susyhit_dir)
    os.system("./run >> " + scripts_dir + "/temp.out")
    
    # Get particle masses from susyhit output
    with open("susyhit_slha.out", "r") as f:
        s = f.read()
        x2 = getParamValue(s, "~chi_20")
        x1 = getParamValue(s, "~chi_10")
        
    os.chdir(scripts_dir)
    return np.array([np.abs(x1), np.abs(x2)])
        

def run_suspect3(timeout=3):
    to_copy = ["M_1", "M_2", "tanbeta(MZ)", "mu(EWSB)", "M_eL" , "M_eR", "M_muL", "M_muR", "M_tauL", "M_tauR"]

    with open(susyhit_dir + "/suspect2_lha.in") as f:
        s = f.read()
    with open(suspect_dir + "/suspect3.in") as f:
        s3 = f.read()
        
    for var in to_copy:
        s3 = changeParamValueStr(s3, var, getParamValue(s, var))

    t0 = time.time()
    while "lock" in os.listdir(suspect_dir):
        if time.time() - t0 > timeout:
            raise NameError("Suspect3 timed out. Lock file still in suspect3 directory. Other thread failed?")

    with open("%s/lock" % suspect_dir, "w") as f:
        pass
        
    with open("%s/suspect3.in" % suspect_dir, "w") as f:
        f.write(s3)

    result = subprocess.run(['%s/suspect3' % suspect_dir, '-d', '%s/suspect3.in' % suspect_dir], stdout=subprocess.PIPE)
    os.system("rm %s/lock" % suspect_dir)
    
    return result.stdout.decode('ascii')

def getMassSplit(M1, M2, mu):
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)
    changeParamValue("mu(EWSB)", mu)

    s = run_suspect3()
    return getParamValue(s, "~chi_20") - abs(getParamValue(s, "~chi_10"))


def correct_neutralinoMassDiff(m1, M2_0, dm=0.1):
    pts = np.array([[m1+dm, m1, m1-dm], [M2_0, M2_0, M2_0]]).T
    dic = run(pts, True, working_directory="test", run_micromegas=False, verbose=False)

    mass_diff = np.abs(dic["~chi_20"]) - np.abs(dic["~chi_10"])
    
    if mass_diff[0] > mass_diff[1] and mass_diff[2] > mass_diff[1]:
        to_break = True

    deriv = (mass_diff[2] - mass_diff[0])/(2*dm)
    deriv2 = (mass_diff[2] - 2*mass_diff[1] + mass_diff[0]) / dm**2

    dm1 = mass_diff[1]/deriv
    if deriv2 != 0:
        if abs(deriv/deriv2) < abs(dm1):
            dm1 = deriv/deriv2

    return [m1 + dm1, M2_0]

def minimize_est(x, y):
    pass


def minimize_neutralinoMassDiff(M1_0, M2_0, tolerance=0.1, stepfactor=1, return_diff=False, verbose=False, return_chain=False, refine=True):

    m1 = M1_0
    dm = tolerance
    
    chain = []
    c = 0

    to_break = False
    while c < 10 and not to_break:
        pts = np.array([[m1+dm, m1, m1-dm], [M2_0, M2_0, M2_0]]).T

        mass_diff = []
        for p in pts:
            changeParamValue("M_1", p[0])
            changeParamValue("M_2", p[1])
            s = run_suspect3()

            mass_diff.append(abs(getParamValue(s, "~chi_20")) - abs(getParamValue(s, "~chi_10")))

        chain.append([m1, mass_diff])

        if mass_diff[0] > mass_diff[1] and mass_diff[2] > mass_diff[1]:
            to_break = True

        if mass_diff[1] < tolerance:
            to_break = True

        deriv = (mass_diff[2] - mass_diff[0])/(2*dm)
        deriv2 = (mass_diff[2] - 2*mass_diff[1] + mass_diff[0]) / dm**2

        dm1 = mass_diff[1]/deriv
        if deriv2 != 0:
            if abs(deriv/deriv2) < abs(dm1):
                dm1 = deriv/deriv2

        dm1 *= stepfactor

        if verbose:
            print("Run %i: \tM1: %.2f \tMass Splitting: %.5f\t" % (c, m1, mass_diff[1]))
        m1 += dm1
        c += 1
        if np.abs(m1) > M2_0 + 100: 
            m1 = np.nan
            break


    corr = 1.
    if not np.isnan(m1) and refine:
        if verbose:
            print("Final Correction With susyhit.")
        m1_sh, _ = correct_neutralinoMassDiff(m1, M2_0)
        corr = m1_sh / m1
        m1 = m1_sh
        if verbose: print("Done")

    if return_chain:
        return chain
    
    if return_diff:
        return [m1, M2_0], mass_diff[1] + deriv * dm1, corr
    else:
        return [m1, M2_0], corr



def old_minimize_neutralinoMassDiff(M1_0, M2_0, tolerance=0.1, stepfactor=0.8, return_diff=False, verbose=False, return_chain=False):

    m1 = M1_0
    dm = tolerance
    
    chain = []
    c = 0

    to_break = False
    while c < 10 and not to_break:
        pts = np.array([[m1+dm, m1, m1-dm], [M2_0, M2_0, M2_0]]).T
        dic = run(pts, True, working_directory="test", run_micromegas=False, verbose=False)

        mass_diff = np.abs(dic["~chi_20"]) - np.abs(dic["~chi_10"])
        

        chain.append([m1, mass_diff])

        if mass_diff[0] > mass_diff[1] and mass_diff[2] > mass_diff[1]:
            to_break = True

        deriv = (mass_diff[2] - mass_diff[0])/(2*dm)
        deriv2 = (mass_diff[2] - 2*mass_diff[1] + mass_diff[0]) / dm**2

        dm1 = mass_diff[1]/deriv
        if deriv2 != 0:
            if abs(deriv/deriv2) < abs(dm1):
                dm1 = deriv/deriv2

        dm1 *= stepfactor

        if verbose:
            print("Run %i: \tM1: %.2f \tMass Splitting: %.3f\t" % (c, m1, mass_diff[1]))
        m1 += dm1
        c += 1
        if np.abs(m1) > M2_0 + 100: 
            m1 = np.nan
            break
    
    if return_diff:
        return [m1, M2_0], mass_diff[1] + deriv * dm1
    else:
        return [m1, M2_0]
    

def optimize_relicDensity(M1_0, M2, mu, m_sleptons, tanB, verbose=False, tolerance=0.01, maxiter=10, save_dd_pval=False):

    changeParamValue("mu(EWSB)", mu)
    changeParamValue("tanbeta(MZ)", tanB)
    changeParamValue("M_eL",   m_sleptons)
    changeParamValue("M_eR",   m_sleptons)
    changeParamValue("M_muL",  m_sleptons)
    changeParamValue("M_muR",  m_sleptons)
    changeParamValue("M_tauL",  m_sleptons)
    changeParamValue("M_tauR",  m_sleptons)
    
    def dOm_dM1(M1, M2, stepsize=0.1):
        dic = run([[M1, M2],[M1+np.sign(M1)*stepsize, M2]], True, 
                  working_directory="test", dd_pval=save_dd_pval, label_by="rand", verbose=False)

        om = dic["omega_dm"]
        deriv = np.sign(M1) * (om[1] - om[0]) / stepsize

        return om[0], deriv, dic
    
    target = 0.11
    max_stepsize = 10

    p0 = [M1_0, M2]
    p = [p0]

    om, deriv, dic = dOm_dM1(*p[-1])

    c = 0
    while np.abs(om - target) > tolerance and c < maxiter:
        dM1 = (0.11 - om) / deriv

        delta = max_stepsize * np.tanh(dM1/max_stepsize)

        p.append([p[-1][0] + delta, p[-1][1]])

        om, deriv, dic = dOm_dM1(*p[-1])
        c += 1
    
    if c == maxiter:
        print("Minimum possibly not found: Omega = %.3e, returning NaN" % om)
        p[-1][0] = np.nan
    
    for k in dic.keys():
        dic[k] = [dic[k][1]]
    
    return p[-1][0], dic


def optimize_relicDensity(M1_0, M2, mu, m_sleptons, tanB, verbose=False, tolerance=0.01, 
                          maxiter=10, save_dd_pval=False):

    if verbose: print("Setting mu,tanB,m_sleptons: %i, %i, %i" % (mu, tanB, m_sleptons))
    changeParamValue("mu(EWSB)", mu)
    changeParamValue("tanbeta(MZ)", tanB)
    changeParamValue("M_eL",   m_sleptons)
    changeParamValue("M_eR",   m_sleptons)
    changeParamValue("M_muL",  m_sleptons)
    changeParamValue("M_muR",  m_sleptons)
    changeParamValue("M_tauL",  m_sleptons)
    changeParamValue("M_tauR",  m_sleptons)
    
    def dOm_dM1(M1, M2, stepsize=0.1):
        dic = run([[M1, M2],[M1+np.sign(M1)*stepsize, M2]], True, 
                  working_directory="test", label_by="rand", verbose=False)

        om = dic["omega_dm"]
        deriv = np.sign(M1) * (om[1] - om[0]) / stepsize

        return om[0], deriv, dic
    
    if mu < abs(M1_0):
        M1_0 = np.sign(M1_0)*mu
    
    l = [[M1_0-np.sign(M1_0)*10*i, M2] for i in range(6)]
    dic = run(l, True, working_directory="test", label_by="rand", verbose=False)

    target = 0.12
    om = np.array(dic["omega_dm"])[np.argsort(dic["M1"])]
    m1 = np.sort(dic["M1"])
    guess = np.interp(target, om, m1)
    
    if verbose:
        print("Initial Range: ")
        print(" ", m1)
        print(" ", om)
    
    max_stepsize = 10

    p0 = [guess, M2]
    p = [p0]

    om, deriv, dic = dOm_dM1(*p[-1])

    c = 0
    while np.abs(om - target) > tolerance and c < maxiter:
        
        if verbose:
            print(p[-1], om, deriv)
        dM1 = (target - om) / deriv

        delta = max_stepsize * np.tanh(dM1/max_stepsize)

        p.append([p[-1][0] + delta, p[-1][1]])

        om, deriv, dic = dOm_dM1(*p[-1])
        c += 1
    
    if c == maxiter:
        print("Minimum possibly not found: Omega = %.3e, returning NaN" % om)
        p[-1][0] = np.nan
    
    for k in dic.keys():
        dic[k] = [dic[k][1]]
    
    return p[-1][0], dic
    
def get_approxGm2(M1, M2, mu, m_sl, tanB):
    
    alpha = 1./137
    sw_sq = 0.25

    m_mu = 0.1
        
    def fxp(x):
        z0 = np.zeros(len(x))
        sel = np.abs(x-1) > 0.01
        z0[sel] = (x[sel]**2 - 4*x[sel] + 3 + 2*np.log(x[sel])) / (1-x[sel])**3
        z0[~sel] = - 2./3 + 0.5*(x[~sel]-1)
        return z0
     
    def fx0(x):
        z0 = np.zeros(len(x))
        sel = np.abs(x-1) > 0.01
        z0[sel] = (x[sel]**2 - 1 - 2*x[sel]*np.log(x[sel])) / (1-x[sel])**3
        z0[~sel] = - 1./3 + 1./6*(x[~sel]-1)
        return z0

    def df_dx(x):
        z0 = np.zeros(len(x))
        sel = np.abs(x-1) > 0.01
        z0[sel] = x[sel]/(1-x[sel])**3 * (2*x[sel] - 2*(1 + np.log(x[sel])) + 3 * (1-x[sel])**2 * fx0(x[sel]))
        z0[~sel] = 1./6 * (1 - 0.2 * (x[~sel] - 1))
        return z0
    
    if not isinstance(m_sl, np.ndarray):
        m_sl=np.array([m_sl])

    a_x0 = -alpha * M1 * m_mu**2 * mu * tanB / (4*np.pi*(1-sw_sq)*m_sl**4) * (fx0((M1/m_sl)**2) + df_dx((M1/m_sl)**2))
    a_xp = alpha*m_mu**2 * mu * M2 * tanB / (4*np.pi*sw_sq*m_sl**2) * (fxp((M2/m_sl)**2) - fxp((mu/m_sl)**2)) / (M2**2 - mu**2)
    
    return a_x0 + a_xp
    
def optimize_gm2(M1, M2, mu, tanB, m_sleptons, test_range=None, N=300, to_minimize="mu", 
                 verbose=False, target=2.5e-9, sigma_tolerance=1, which="low", use_micromegas=True):

    changeParamValue("mu(EWSB)", mu)
    changeParamValue("tanbeta(MZ)", tanB)        
        
    if test_range == None:
        test_range = [M2+1, 3005.]
        
    def get_gm2(M1, M2, mu, m_sleptons):
        
        changeParamValue("mu(EWSB)", mu)
        
        changeParamValue("M_eL",   m_sleptons)
        changeParamValue("M_eR",   m_sleptons)
        changeParamValue("M_muL",  m_sleptons)
        changeParamValue("M_muR",  m_sleptons)
        changeParamValue("M_tauL", m_sleptons)
        changeParamValue("M_tauR", m_sleptons)
        #changeParamValue("M_tauL", 2.5e3)
        #changeParamValue("M_tauR", 2.5e3)
        
        queue = mp.Queue()
        run_once(M1, M2, True, queue, working_directory="test")
        return queue.get()["gm2"]  
    
    if to_minimize == "sleptons":
        xtest = np.logspace(*np.log10(test_range), N)
        gm2_est = get_approxGm2(M1, M2, mu, xtest, tanB)
        gm2_susyhit = lambda x: get_gm2(M1, M2, mu, x)
        
    elif to_minimize == "mu":
        xtest = np.logspace(*np.log10(test_range), N)
        gm2_est = get_approxGm2(M1, M2, xtest, m_sleptons, tanB)
        gm2_susyhit = lambda x: get_gm2(M1, M2, x, m_sleptons)

    def get_best(xtest, gm2_est):
        
        score = (gm2_est-target)*1e9

        candidates = []
        for i in range(0, len(xtest)-1):
            if not np.isnan(score[i]) and not np.isnan(score[i+1]):
                if np.sign(score[i]) != np.sign(score[i+1]):
                    direc = int(np.sign(score[i+1]))
                    candidates.append(np.interp(0, score[i:i+2][::direc], xtest[i:i+2][::direc]))

        if len(candidates) > 0:
            if len(candidates) > 1:
                print("Multiple gm2 candidates found: %s" % (str(candidates)))
                pass
            if which=="low":
                return np.min(candidates), target
            elif which=="high":
                return np.max(candidates), target
            else:
                raise NameError("Which must be \"high\" or \"low\"")
        
        else:
            cand = np.argmin(np.abs(score))
            if np.abs(score[cand]) < 0.59*sigma_tolerance:
                return xtest[cand], gm2_est[cand]
            else:
                return np.nan, np.nan
    
    initial, gm2_init = get_best(xtest, gm2_est)
        
      
    if np.isnan(initial): 
        print("Warning: Unable to satisfy g-2 with these parameters: (M1, M2, mu, tanB) = (%i, %i, %i, %i)" % (M1, M2, mu, tanB))
        
        ind = np.argmin(np.abs(gm2_est-target)*1e9)
        print("\t Closest point: (m_sleptons, delta-g) = (%i, %.2E)" % (xtest[ind], gm2_est[ind]))
        return np.nan
    
    if use_micromegas == True:
        gm2_true = gm2_susyhit(initial)
        correction_size = gm2_true / gm2_init

        final, gm2_corr = get_best(xtest, correction_size * gm2_est)

        gm2_final = gm2_susyhit(final)
    else:
        final = initial
    
    if verbose:
        print("Tested %s range: \t%i - %i" % (to_minimize, min(xtest), max(xtest)))
        print("Tree level gm2 range: \t%.2E - %.2E" % (min(gm2_est), max(gm2_est)))
        print("Loop Correction Size: \t%.2f" % ((correction_size - 1) * 100) + "%")
        print("Initial %s, gm2 values: \t%i , %.2E" % (to_minimize, initial, gm2_true))
        print("Final %s, gm2 values: \t%i , %.2E" % (to_minimize, final, gm2_final))

    
    return final

def old_optimize_gm2_array(points, m_sleptons, tanB, verbose=False, sigma_target=0, sigma_tolerance=1, use_micromegas=True):
    
    delta_amu = 2.51e-9
    sigma = 0.59e-9
    
    target = delta_amu + sigma_target*sigma

    changeParamValue("M_eL",   m_sleptons)
    changeParamValue("M_eR",   m_sleptons)
    changeParamValue("M_muL",  m_sleptons)
    changeParamValue("M_muR",  m_sleptons)
    changeParamValue("M_tauL", m_sleptons)
    changeParamValue("M_tauR", m_sleptons)   
    changeParamValue("tanbeta(MZ)", tanB)
    
    mu_l = []
    gm2_approx = []
    is_within_tolerance = []
    for p in points:
        f = lambda m: ((get_approxGm2(*p, m, m_sleptons, tanB) - target)*1e9)**2

        fit = minimize(f, [m_sleptons*2], tol=1e-5)
        mu_l.append(fit.x[0])
        gm2_approx.append(get_approxGm2(*p, fit.x[0], m_sleptons, tanB))
        
    def additional_command(M1, M2, index):
        changeParamValue("mu(EWSB)", mu_l[index])
    
    
    if use_micromegas == True:
        sh_gm2 = 0
        #while abs(sh_gm2-target) < sigma_tolerance * sigma:
            
        dic = run(points, True, working_directory="./temp", additional_command=additional_command, verbose=verbose)

        mu_corr = []
        gm2_corr = []

        for i,p in enumerate(points):
            correction = dic["gm2"][i] / gm2_approx[i]
            fc = lambda m: ((correction*get_approxGm2(*p, m, m_sleptons, tanB) - target)*1e9)**2

            fitc = minimize(fc, [m_sleptons*2])
            print(mu_l[i], fitc.x[0], correction)

            mu_corr.append(fitc.x[0])
            gm2_corr.append(correction*get_approxGm2(*p, fitc.x[0], m_sleptons, tanB))
            #print(gm2_corr)
            
        return mu_corr, gm2_corr
    else:
        return mu_l, gm2_approx

def optimize_gm2_array(points, m_sleptons, tanB, verbose=False, sigma_target=0., sigma_tolerance=1, use_micromegas=True, cap=1e10):
    
    delta_amu = 2.51e-9
    sigma = 0.59e-9
    
    target = delta_amu + sigma_target*sigma

    changeParamValue("M_eL",   m_sleptons)
    changeParamValue("M_eR",   m_sleptons)
    changeParamValue("M_muL",  m_sleptons)
    changeParamValue("M_muR",  m_sleptons)
    changeParamValue("M_tauL", m_sleptons)
    changeParamValue("M_tauR", m_sleptons)   
    changeParamValue("tanbeta(MZ)", tanB)     
    
    def minim(points, mu_l, corr):
        mu_corr = []
        gm2 = []
        for i,p in enumerate(points):
            f = lambda m: 10*((corr[i]*get_approxGm2(*p, m*100, m_sleptons, tanB) - target)*1e9)**2

            fit = minimize(f, [mu_l[i]/100], tol=1e-5)
            mu_corr.append(fit.x[0]*100)
            gm2.append(corr[i]*get_approxGm2(*p, fit.x[0]*100, m_sleptons, tanB)[0])
        return mu_corr, gm2
        
    correction = np.ones(len(points))
    mu_l = np.array(points).T[1]+10 #np.ones(len(points)) * m_sleptons 
    correctAgain = np.ones(len(points), dtype=bool)

    def additional_command(M1, M2, index):
        changeParamValue("mu(EWSB)", np.array(mu_l)[correctAgain][index])

    c = 0
    while np.any(correctAgain) and c < 5:
        print("Correcting %i points with susyhit/micromegas:" % sum(correctAgain)) 
        mu_l, gm2 = minim(points, mu_l, correction)

        if not use_micromegas:
            break
            
        dic = run(np.array(points)[correctAgain], True, working_directory="./gm2fixing", 
                  additional_command=additional_command, verbose=verbose, label_by="index")
        print()

        correction[correctAgain] *= np.array(dic["gm2"]) / np.array(gm2)[correctAgain]
        correctAgain[correctAgain] = np.abs(np.array(dic["gm2"]) - target) > sigma_tolerance * sigma
        c += 1
    
    for i in range(len(mu_l)):
        if mu_l[i] > cap:
            mu_l[i] = cap
    return mu_l

def logf(string, file):
    if os.path.isfile(file):
        with open(file, "w") as f:
            f.write("%s\n" % string)
    else:
        try: 
            with open(file, "x") as f:
                f.write("%s\n" % string)
        except:
            pass

def lock(direc, timeout=10):
    t0 = time.time()
    while "lock" in os.listdir(susyhit_dir):
        if time.time() - t0 > timeout:
            raise NameError("%s is locked, check if there is a lock file in it." % direc)

    open("%s/lock" % direc, 'a').close()

def unlock(direc):
    os.system("rm %s/lock" % direc)


def run_once(M1, M2, remake, out_queue, run_prospino=False, 
             run_micromegas=True, run_checkmate=False, working_directory=".", 
             additional_command=None, index=0, label_by=None, timeout=5):
    """
    Runs susyhit, micromegas, checkmate, and/or prospino once. Each program can be toggled.
    """
    masses_to_save = ["~chi_10", "~chi_20"]
    
    logfile = working_directory + "/logs/%i.log" % index

    tag = "%i_%i" % (M1, M2)
    #if label_by == "full":
    #    for k in additional_settings.keys():
    #        #tag += "_%i" % additional_settings[k]
    
    if label_by == "rand":
        tag = str(time.time()).split(".")[1]

    elif label_by == "ordered" or label_by == "index":
        tag = str(index).zfill(6)
        
    # Set up
    filename = "spectrum_%s.dat" % tag
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    for subdir in ["spectra_slha", "prospino_cx", "checkmate", "micromegas_out", "logs"]:
        if subdir not in os.listdir(working_directory):
            os.mkdir(working_directory + "/" + subdir)

    if os.path.isfile(logfile):
        os.system("rm -f %s" % logfile)
        
    log = open(logfile, "x")
    cwd = os.getcwd()
    
    if isinstance(remake, bool):
        remake = {"prospino": remake, "susyhit": remake, "checkmate": remake}

    lock(susyhit_dir)

    # Set param values to run over
    changeParamValue("M_1", M1)
    changeParamValue("M_2", M2)
    if additional_command != None:
        additional_command(M1, M2, index)
    
    log.write("Running susy-hit...\n")
    # Run susyhit
    if filename not in os.listdir(working_directory + "/spectra_slha") or remake["susyhit"]:
        os.chdir(susyhit_dir)
        unlock(susyhit_dir)
        os.system("./run >> " + scripts_dir + "/temp.out")
        os.chdir(cwd)
        os.system("cp " + susyhit_dir + "/susyhit_slha.out " + working_directory + "/spectra_slha/" + filename)
        
    # Get particle masses from susyhit output
    with open(working_directory + "/spectra_slha/" + filename, "r") as f:
        s = f.read()
        if np.isnan(getParamValue(s, "EWSB")):
            out = {"index": index, "M1": M1, "M2": M2, "cx": np.nan, "r": np.nan, "analysis": np.nan, "tag": tag}
            for var in ["gm2", "omega_dm", "cx_SI", "cx_SDn", "cx_SDp"] + masses_to_save:
                out[var] = np.nan
                
            out_queue.put(out)
            return None
              
        else:
            # Get particle masses from susyhit output
            outdir = {"index": index, "M1": M1, "M2": M2}
            for label in masses_to_save:
                outdir[label] = getParamValue(s, label)
        
        
    log.write("Running micromegas... \n")
    # Calculate g-2 contribution using micromegas
    dd_om_excluded = False
    if run_micromegas:
        entries = ["gm2", "omega_dm", "cx_SI", "cx_SDp", "cx_SDn"]
        labels = ["g-2 contribution", "Omega_DM", "max SI cx", "SDp cx", "SDn cx"]
        s = os.popen(micromegas_dir + '/MSSM/get_gm2 ./' + working_directory + '/spectra_slha/' + filename).read()
        
        log.write("\n Micromegas Output: \n%s\n" % s)
        lines = s.split("\n")[:-1]

        for l in lines:
            isIn = [lab in l for lab in labels]
            if sum(isIn) > 0:
                idx = np.argmax(isIn)
                outdir[entries[idx]] = float(l.split(": ")[1])
            
        for e in entries:
            if e not in outdir.keys():
                print("Couldn't get %s from: %s" % (e, lines))
                outdir[e] = np.nan
           
        with open(working_directory + "/micromegas_out/micromegas_%i_%i.dat" % (M1, M2), "w") as f:
            f.write(s)
                            
    if run_checkmate and not dd_om_excluded:
        outname = "checkmate_%s.dat" % tag
        
        if outname not in os.listdir(working_directory + "/checkmate") or remake["checkmate"]:
            os.system("cp " + working_directory + 
                      "/spectra_slha/" + filename + 
                      " " + checkmate_dir + "/bin/")
            
            os.chdir(checkmate_dir + "/bin")
            os.system("python runCheckmate.py " + filename)
            
            os.system("rm " + filename)
            os.chdir(cwd)
           
            os.system("cp " + checkmate_dir + "/results/" + filename.split(".")[0] + "/result.txt " + working_directory + "/checkmate/" + outname)
            os.system("cp -r " + checkmate_dir + "/results/" + filename.split(".")[0] + "/evaluation " + working_directory + "/checkmate/" + "evaluation_%s" % tag)

            #os.system("rm -rf ../results/" + outname)
            
        # Read checkmate output and get r value
        with open(working_directory + "/checkmate/" + outname) as f:
            s=f.readlines()
            r = np.nan
            for l in s:
                line = l.split(": ")
                if line[0] == "Result for r":
                    r = float(line[1])
                if line[0] == "Analysis":
                    analysis = line[1]

    else:    
        r = np.nan
        analysis = ""
    
    outdir["tag"] = tag
    outdir["r"] = r 
    outdir["analysis"] = analysis

    out_queue.put(outdir)
    
    log.write("Done!") 
    log.close()
    return None
    

def run(points_list, remake, run_prospino=False, run_micromegas=True, 
        run_checkmate=False,
        working_directory=None, verbose=True, additional_command=None, n_procs=6,
        label_by=None, match_input=True):
    
    """
    Over a list of [[M1, M2], [M1, M2], ...], runs susy programs and gives the output as a queue.
    Uses multiprocessing to speed up results.
    """
    
    if isinstance(remake, bool):
        remake = {"prospino": remake, "susyhit": remake, "checkmate": remake}
        
    if working_directory == None:
        working_directory = "output"
    
    # Set up directories
    Path(working_directory).mkdir(parents=True, exist_ok=True)
        
    for subdir in ["spectra_slha", "prospino_cx", "checkmate", "micromegas_out", "logs"]:
        if subdir not in os.listdir(working_directory):
            os.mkdir(working_directory + "/" + subdir)    
        
    processors = []

    out = mp.Queue()

    # Create processes, 3 at a time, to be run by cpus
    t0 = time.time()
    
    if verbose:
        print("Running %i points... " % len(points_list))
    n_completed = 0
    i = 0
    
    while out.qsize() < len(points_list):

                
        if len(processors) - out.qsize() < n_procs and len(processors) < len(points_list):
            proc = mp.Process(target=run_once, args=(points_list[i][0], points_list[i][1], remake, out, 
                                                run_prospino, run_micromegas,
                                                run_checkmate,
                                                working_directory, additional_command,
                                                i, label_by, True))

            proc.start()
            processors.append(proc)
            i += 1
            
        if verbose:
            runtime = time.time() - t0
            print("Process %i/%i Done! \t Runtime: %s\r" % (out.qsize()+1, len(points_list), 
                         str(datetime.timedelta(seconds=int(runtime)))), end="")
        time.sleep(0.1)
        
    data = {}
    ret = out.get()
    for k in ret.keys():
        data[k] = [ret[k]]
        
    while out.qsize() > 0:
        ret = out.get()
        for k in ret.keys():
            data[k].append(ret[k])
    
    if match_input:
        indsort = np.argsort(data["index"])
        for k in data.keys():
            data[k] = [data[k][i] for i in indsort]
        
    return data
        
    
############ Section of code for generating checkmate run points #################
import scipy.optimize as op


def _get_next_point(boundary, points):
    """
    Gets the next boundary point using gift wrapping algorithm
    https://en.wikipedia.org/wiki/Gift_wrapping_algorithm
    """

    mag3 = lambda x: np.sqrt(np.sum(x**2))

    if len(boundary) < 2:
        return np.nan

    p0 = boundary[-1]
    vec = boundary[-1] - boundary[-2]

    candidate = p0
    cand_cosang = -1.
    for p in points:
        if mag3(p-p0) == 0:
            continue

        cosang = np.dot(vec, p-p0) / mag3(vec) / mag3(p-p0)
        if cosang == cand_cosang:
            if mag3(p - p0) > mag3(candidate - p0):
                candidate = p
                cand_cosang = cosang
        elif cosang > cand_cosang:
            candidate = p
            cand_cosang = cosang

    return candidate


def get_boundary(points):
    """
    Returns the boundary of a set of points
    """
    mag3 = lambda x: np.sqrt(np.sum(x**2))

    p0 = points[np.argmin(points.T[0])]
    for p in points:
        if p[0] == p0[0] and p[1] > p0[1]:
            p0 = p

    p1 = _get_next_point(np.array([np.array([p0[0], p0[1] - 1]), p0]), points)
    boundary = [p0, p1]

    c = 0

    while mag3(boundary[-1] - boundary[0]) > 0  and c < 100:
        c += 1
        boundary.append(_get_next_point(boundary, points))

    return np.vstack(boundary)

def costFunction(points, boundary, boundaryfactor = 1., widths=np.array([1.,1.]), totalcharge=1.):
    """ 
    A cost function which can be minimized with scipy to find the optimal point layout within 
    a set of boundary points.
    """
    n = len(points)
    
    distances = np.hstack([np.sqrt(np.sum(((points[j:] - points[:-j])/widths)**2, axis=1)) 
                           for j in range(1, len(points))])
    
    potential = totalcharge / n * np.sum(1./distances**2)
    return potential / n
    
def keep_insideboundary(boundary, r):
    
    if Polygon(boundary).contains(Point(r)):
        return r
    else:
        if np.any(boundary[0] != boundary[-1]):
            boundary = np.vstack((boundary, boundary[0]))

        dr = boundary[1:] - boundary[:-1]
        dr_mag = np.sqrt(np.sum(dr**2, axis=1))
        tval = np.sum((boundary[:-1] - r) * dr, axis = 1) / dr_mag**2

        tval[tval > 1.] = 1.
        tval[tval < 0] = 0

        dist = np.sqrt(np.sum(( dr * tval[:,None] + boundary[:-1] - r )**2, axis=1))
        idx = np.argmin(dist)
        return (dr * tval[:,None] + boundary[:-1])[idx]
    
def optimize_points(r0, boundary, cost_function, n_steps=50, return_full=False):
    
    dx_mag = 0.001
    stepsize = 0.02
    f = lambda r: costFunction(r, boundary)

    xp = [r0]

    cost = [f(r0)]
    minfound=False
    for step in range(n_steps):
        if minfound: 
            continue
            
        dc_dxi_0 = []
        for i in range(len(xp[-1])):
            dc_dxi_0.append([])
            for j in range(2):

                dx = np.zeros(xp[0].shape)
                dx[i, j] = dx_mag*stepsize
                dc_dxi_0[i].append( (f(xp[-1] + dx) - f(xp[-1]))/(dx_mag*stepsize))

        grad = np.array(dc_dxi_0)
        
        mags = np.sum(grad, axis=1)
        toobig = stepsize*mags > 0.05
        grad[toobig] = grad[toobig] / mags[toobig][:,None] * 0.05
        proposed = np.array([keep_insideboundary(boundary, r) for r in xp[-1] - stepsize * grad])
        
        co = f(proposed)
        
        if not minfound:
            xp.append(proposed)
            cost.append(f(xp[-1]))   
                
    if return_full:
        return xp
    else:
        return xp[-1]

    
def get_allowed_polygon(x, y, om, dd):
    """
    Generates the boundary points given the omega_dm and direct detection constraints.
    """
    com = plt.tricontourf(x,y,om, levels=[0.1, 1e10], alpha=0);
    om_bounds = [col.get_paths()[0].vertices.T for col in com.collections][0]
    
    cdm = plt.tricontourf(x,y,dd, levels=[-1,0.05], alpha=0);
    dd_bounds = [col.get_paths()[0].vertices.T for col in cdm.collections][0]

    mp = MultiPoint(np.array([x, y]).T)
    p_space = mp.buffer(1).buffer(-1)

    if len(dd_bounds.T) > 2:
        dd_pol = Polygon(dd_bounds.T)
        p_space = p_space.difference(dd_pol)
    if len(om_bounds.T) > 2:
        om_pol = Polygon(om_bounds.T)
        p_space = p_space.difference(om_pol)
    
    scan_region = p_space.buffer(-0.01).buffer(0.01)
    
    boundary_points = [scan_region.boundary.interpolate(a, normalized=True) for a in np.linspace(0, 1, 100)]
    boundary_coords = np.array([[b.x, b.y] for b in boundary_points])
    return boundary_coords

def generate_grid(number, boundary):
    """
    Generates a grid of N points within the boundary. Generally a good initial guess for minimization.
    """
    xmax, ymax = np.max(boundary, axis=0)*0.99
    xmin, ymin = np.min(boundary, axis=0)*1.01

    wx = xmax - xmin
    wy = ymax - ymin

    density = number / (wx * wy)
    area = Polygon(boundary).area

    Np = number * number/(area*density)
    N_axis = int(np.ceil(np.sqrt(Np)*1.2))

    n_inside = 10000
    
    while n_inside > number:
        g0 = np.meshgrid(np.linspace(xmin, xmax, N_axis), np.linspace(ymin, ymax, N_axis))
        p0 = np.vstack((g0[0].flatten(), g0[1].flatten()))

        n_inside = np.sum([Polygon(boundary).contains(Point(p)) for p in p0.T])
        xmax *= 1.005
        ymax *= 1.005

    p_out = []
    for p in p0.T:
        if Polygon(boundary).contains(Point(p)):
            p_out.append(p)

    return np.array(p_out)

def get_checkmatePoints(x, y, om, dd, N, totalcharge=1., return_full=False, n_steps=50):
    
    wx = np.max(x) - np.min(x)
    wy = np.max(y) - np.min(y)
    boundary = get_allowed_polygon(x/wx, y/wy, om, dd)

    plt.close()

    grid_points = generate_grid(N, boundary)
    widths = np.max(boundary, axis=0) - np.min(boundary, axis=0)
    scan_points = optimize_points(grid_points, boundary, costFunction, n_steps=n_steps, 
                                  return_full=return_full)

    return np.array(scan_points)*np.array([wx,wy]), boundary* np.array([wx, wy])


