{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "265c0a4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import h5py\n",
    "import matplotlib.pyplot as plt\n",
    "import scipy\n",
    "from scipy import linalg\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "import time\n",
    "import pickle\n",
    "import sys\n",
    "import itertools\n",
    "import math\n",
    "import matplotlib as mpl\n",
    "from mpl_toolkits.mplot3d import Axes3D\n",
    "from matplotlib.ticker import MaxNLocator\n",
    "from scipy.stats import wasserstein_distance\n",
    "from scipy.stats import entropy\n",
    "plt.rcParams['figure.facecolor'] = 'white'\n",
    "plt.rcParams[\"figure.facecolor\"] = \"w\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e63b76d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def hist0d(xdata,hbins,geom):\n",
    "\n",
    "    x = xdata\n",
    "    # Define the borders\n",
    "    xmin = min(x)# - deltaX\n",
    "    xmax = max(x)# + deltaX\n",
    "    if geom==1:\n",
    "        binsar=np.geomspace(1e-2, xmax, num=len(hbins))\n",
    "    else:\n",
    "        binsar=hbins\n",
    "    histx, binsx = np.histogram(x, bins=binsar, density=True)\n",
    "    xm = 0.5*(binsx[1:] + binsx[:-1])\n",
    "\n",
    "    return xm, histx"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6218f100",
   "metadata": {},
   "outputs": [],
   "source": [
    "def timeseriesdot(x,y, multype): \n",
    "    tsdot = np.einsum(multype,x,y.T) #Einstein summation. Index i is time.\n",
    "    return tsdot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d54a5e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "def makePDF(x, bins, N, geom):    \n",
    "    binsz = np.shape(bins)[0] - 1    \n",
    "    \n",
    "    if N>1:\n",
    "        PDF = np.zeros((binsz,N,2))\n",
    "        for cly in range(N):\n",
    "            if system=='cdv' and cly==0:\n",
    "                PDF[:,cly,0], PDF[:,cly,1] = hist0d(x[:,cly],bins[:,cly],geom) #For CLVs theta PDF, set geom=1\n",
    "            else:                \n",
    "                PDF[:,cly,0], PDF[:,cly,1] = hist0d(x[:,cly],bins[:,cly],geom)\n",
    "    else:\n",
    "        PDF = np.zeros((binsz,2))\n",
    "        PDF[:,0], PDF[:,1] = hist0d(x,bins,geom)\n",
    "    return PDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "41888ced",
   "metadata": {},
   "outputs": [],
   "source": [
    "def timerange(tot, frac):\n",
    "    inig = int(tot/frac)\n",
    "    fing = tot - inig\n",
    "    return inig, fing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63e11ee6",
   "metadata": {},
   "source": [
    "## Statistics of the Instantaneous Covariant Lyapunov exponents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b50c93dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "f1 = h5py.File('/Users/eo821/Documents/PhD_Research/PI-LSTM/Lorenz_LSTM/src/trainings/L96/D-20/cool-sweep-2/16500_clvs.h5','r+')\n",
    "f1.keys()\n",
    "np.array(f1.get('thetas_clv')).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fcfc4b75",
   "metadata": {},
   "outputs": [],
   "source": [
    "f2 = h5py.File('/Users/eo821/Documents/PhD_Research/PI-LSTM/ESN_CLVs/GM/lorenz96/target_data_N_20/CLV_results/ESN_target_CLV_dt_0.01_noise_0.h5','r+')\n",
    "f2.keys()\n",
    "np.array(f2.get('thetas_clv')).shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "10c92265",
   "metadata": {},
   "outputs": [],
   "source": [
    "FTCLE_lstm = np.array(f1.get('thetas_clv')).T\n",
    "FTCLE_targ = np.array(f2.get('thetas_clv')).T\n",
    "N_max = min(FTCLE_lstm.shape[1], FTCLE_targ.shape[1])\n",
    "FTCLE_lstm = FTCLE_lstm[:, :N_max]\n",
    "FTCLE_targ = FTCLE_targ[:, :N_max]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68cab776",
   "metadata": {},
   "outputs": [],
   "source": [
    "system='lorenz96'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bdcb9e76",
   "metadata": {},
   "outputs": [],
   "source": [
    "total_length_lstm = np.shape(FTCLE_lstm)[1] # choose the correct time index.\n",
    "total_length_targ = np.shape(FTCLE_targ)[1] # choose the correct time index.\n",
    "\n",
    "inig_lstm, fing_lstm = timerange(total_length_lstm,10)\n",
    "inig_targ, fing_targ = timerange(total_length_targ,10)\n",
    "\n",
    "binsz = 40\n",
    "if system == 'lorenz96' or system=='cdv':\n",
    "    geom  = 0 #if 1: use geometric spacing for histogram\n",
    "else:\n",
    "    geom = 0\n",
    "#make histogram\n",
    "ftcles_lstm = FTCLE_lstm[:,inig_lstm:fing_lstm].copy()\n",
    "ftcles_targ = FTCLE_targ[:,inig_targ:fing_targ].copy()\n",
    "\n",
    "xmin = np.amin(ftcles_lstm, axis=(1))\n",
    "xmax = np.amax(ftcles_lstm, axis=(1))\n",
    "binsar=np.linspace(xmin, xmax, num=binsz)\n",
    "            \n",
    "PDF_ftcles_lstm = np.array(makePDF(ftcles_lstm.T, binsar, 3, geom))\n",
    "PDF_ftcles_targ = np.array(makePDF(ftcles_targ.T, binsar, 3, geom))   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd27e167",
   "metadata": {},
   "outputs": [],
   "source": [
    "labels = {0:r'$\\theta_{U,N}$',1:r'$\\theta_{U,S}$',2:r'$\\theta_{N,S}$'}\n",
    "\n",
    "#LSTM\n",
    "PDF_x_lstm = PDF_ftcles_lstm[:,:,0].copy()\n",
    "PDF_y_lstm = PDF_ftcles_lstm[:,:,1].copy()\n",
    "\n",
    "#TARGET\n",
    "PDF_x_targ = PDF_ftcles_targ[:,:,0].copy()\n",
    "PDF_y_targ = PDF_ftcles_targ[:,:,1].copy()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f96da808",
   "metadata": {},
   "outputs": [],
   "source": [
    "Nlyap = 3\n",
    "lntypes = {0:'r-',1:'b-',2:'g-'}\n",
    "lntypes_esn = {0:'r--',1:'r--',2:'r--'}\n",
    "lw = 5\n",
    "fs = 18\n",
    "\n",
    "fig, axs = plt.subplots(1, Nlyap,figsize=(10.5,3.5), sharex=False, sharey=True)\n",
    "\n",
    "ylabelpos = 0.05; xlabelpos = 0.0\n",
    "titles = {0:'(a)',1:'(b)',2:'(c)'}\n",
    "plt.rcParams.update({'font.size': fs})\n",
    "\n",
    "#List the Wasserstein distance\n",
    "wasser_dist_theta = np.zeros(Nlyap)\n",
    "    \n",
    "for cly, ax in zip(range(Nlyap),axs.flat):    \n",
    "    \n",
    "    label = labels[cly]\n",
    "    ltp = lntypes[cly]\n",
    "    if system=='lorenz96' or system=='cdv':\n",
    "        ltp_lstm = 'r--'\n",
    "    else: \n",
    "        ltp_lstm = lntypes_esn[cly]\n",
    "    ax.tick_params(axis='y',pad=0.2)\n",
    "    ax.set_title(titles[cly])\n",
    "    ax.grid(True,c='lightgray',linestyle='--', linewidth=0.5)\n",
    "    ax.set_xlabel(label,fontsize=fs)\n",
    "    \n",
    "    #PLOT TARGET\n",
    "    ax.plot(PDF_x_targ[:,cly], PDF_y_targ[:,cly],'k-',lw=lw,label='Target');\n",
    "        \n",
    "    \n",
    "    #PLOT LSTM\n",
    "    ax.plot(PDF_x_lstm[:,cly], PDF_y_lstm[:,cly],ltp_lstm,lw=lw-1,label='LSTM');\n",
    "        \n",
    "    #set number of ticks\n",
    "    ax.locator_params(tight=True, nbins=4)\n",
    "    \n",
    "    wasser_dist_theta[cly] = wasserstein_distance(PDF_y_lstm[:,cly], PDF_y_targ[:,cly])\n",
    "    print('Wasserstein distance x', cly, ': ', wasser_dist_theta[cly])\n",
    "\n",
    "\n",
    "fig.text(ylabelpos, 0.5, 'PDF',  fontsize=fs+2, ha='center', va='center', rotation='vertical')\n",
    "\n",
    "plt.show()\n",
    "\n",
    "print('average Wasserstein distance:', np.average(wasser_dist_theta))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2470955a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "lorenz_lstm",
   "language": "python",
   "name": "lorenz_lstm"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
