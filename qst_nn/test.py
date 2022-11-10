
#import tensorflow as tf

import numpy as np
import qutip

from qutip import coherent_dm as qutip_coherent_dm
from qutip import thermal_dm as qutip_thermal_dm
from qutip import Qobj, fock, coherent, displace

from qutip.states import fock_dm as qutip_fock_dm
from qutip.states import thermal_dm as qutip_thermal_dm
from qutip.random_objects import rand_dm
from scipy.special import binom
from math import sqrt

import matplotlib.tri as tri
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
import matplotlib.pyplot as plt





def fock_dm(hilbert_size, n=None):
    """
    Generates a random fock state.
    
    Parameters
    ----------
    n : int
        The fock number
    Returns
    -------
    fock_dm: `qutip.Qobj`
        The density matrix as a quantum object.
    """
    if n == None:
        n = np.random.randint(1, hilbert_size/2 + 1)
    return qutip_fock_dm(hilbert_size, n), -1

def plot_fock(rho, title=None, dm_cut=16):
  fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
  inches_per_pt = 1.0/72.27               # Convert pt to inch
  golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
  fig_width = fig_width_pt*inches_per_pt  # width in inches
  fig_height = fig_width*golden_mean      # height in inches
  fig_size =  [fig_width,fig_height]
  params = {# 'backend': 'ps',
            'axes.labelsize': 8,
            'font.size': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'axes.labelpad': 1,
            'text.usetex': False,
            'figure.figsize': fig_size}
  plt.rcParams.update(params)


  fig, ax = plt.subplots(1, 1, figsize=(fig_width/3.27, fig_width/3.4))
  ax.set_xticks([int(dm_cut), int(dm_cut/2), 0])
  ax.set_xticklabels([int(dm_cut), "", 0])

  plt.subplots_adjust(hspace=-.69)

  ax.set_frame_on(True)
  N = rho.shape[0]
  ax.bar(np.arange(0, N), np.real(rho.diag()),
    color="green", alpha=0.6, width=0.8)

  ax.set_xlim(-.5, int(dm_cut))

  ymax = np.max(np.real(rho.diag()))

  ax.set_yticks([0, ymax/2, ymax])

  ax.set_yticklabels([0, '', '{:.1f}'.format(ymax)])

  ax.set_ylabel(r"p(n)", labelpad=-9)
  ax.set_xlabel(r"$|n\rangle$", labelpad=-6)

  fig.subplots_adjust(right=0.85, wspace=0.01, hspace=-7.6)
  ax.set_ylim(0, ymax)
  if title is not None:
    ax.set_title(title, x=0.35)

  # ax.set_aspect('equal')
  return fig, ax

#dm = fock_dm(4,n=None)




#qutip.settings.colorblind_safe = True 
#fig, ax = qutip.hinton(dm,color_style="threshold")
#fig.show()




