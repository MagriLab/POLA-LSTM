import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["figure.facecolor"] = "w"


def hist0d(xdata,hbins,geom):

    x = xdata
    # Define the borders
    xmin = min(x)# - deltaX
    xmax = max(x)# + deltaX
    if geom==1:
        binsar=np.geomspace(1e-2, xmax, num=len(hbins))
    else:
        binsar=hbins
    histx, binsx = np.histogram(x, bins=binsar, density=True)
    xm = 0.5*(binsx[1:] + binsx[:-1])

    return xm, histx

def timeseriesdot(x,y, multype): 
    tsdot = np.einsum(multype,x,y.T) #Einstein summation. Index i is time.
    return tsdot

def makePDF(x, bins, N, geom, system='lorenz96'):    
    binsz = np.shape(bins)[0] - 1    
    
    if N>1:
        PDF = np.zeros((binsz,N,2))
        for cly in range(N):
            if system=='cdv' and cly==0:
                PDF[:,cly,0], PDF[:,cly,1] = hist0d(x[:,cly],bins[:,cly],geom) #For CLVs theta PDF, set geom=1
            else:                
                PDF[:,cly,0], PDF[:,cly,1] = hist0d(x[:,cly],bins[:,cly],geom)
    else:
        PDF = np.zeros((binsz,2))
        PDF[:,0], PDF[:,1] = hist0d(x,bins,geom)
    return PDF

def timerange(tot, frac):
    inig = int(tot/frac)
    fing = tot - inig
    return inig, fing

def plot_clv_pdf(FTCLE_lstm, FTCLE_targ, img_filepath, system='lorenz96'):
    total_length_lstm = np.shape(FTCLE_lstm)[1] # choose the correct time index.
    total_length_targ = np.shape(FTCLE_targ)[1] # choose the correct time index.

    inig_lstm, fing_lstm = timerange(total_length_lstm,10)
    inig_targ, fing_targ = timerange(total_length_targ,10)

    binsz = 40
    if system == 'lorenz96' or system=='cdv':
        geom  = 0 #if 1: use geometric spacing for histogram
    else:
        geom = 0
    #make histogram
    ftcles_lstm = FTCLE_lstm[:,inig_lstm:fing_lstm].copy()
    ftcles_targ = FTCLE_targ[:,inig_targ:fing_targ].copy()

    xmin = np.amin(ftcles_lstm, axis=(1))
    xmax = np.amax(ftcles_lstm, axis=(1))
    binsar=np.linspace(xmin, xmax, num=binsz)
                
    PDF_ftcles_lstm = np.array(makePDF(ftcles_lstm.T, binsar, 3, geom))
    PDF_ftcles_targ = np.array(makePDF(ftcles_targ.T, binsar, 3, geom))
    labels = {0:r'$\theta_{U,N}$',1:r'$\theta_{U,S}$',2:r'$\theta_{N,S}$'}

    #LSTM
    PDF_x_lstm = PDF_ftcles_lstm[:,:,0].copy()
    PDF_y_lstm = PDF_ftcles_lstm[:,:,1].copy()

    #TARGET
    PDF_x_targ = PDF_ftcles_targ[:,:,0].copy()
    PDF_y_targ = PDF_ftcles_targ[:,:,1].copy()
    Nlyap = 3
    lntypes = {0:'r-',1:'b-',2:'g-'}
    lntypes_esn = {0:'r--',1:'r--',2:'r--'}
    lw = 5
    fs = 18

    fig, axs = plt.subplots(1, Nlyap,figsize=(10.5,3.5), sharex=False, sharey=True)

    ylabelpos = 0.05; xlabelpos = 0.0
    titles = {0:'(a)',1:'(b)',2:'(c)'}
    plt.rcParams.update({'font.size': fs})

    #List the Wasserstein distance
    wasser_dist_theta = np.zeros(Nlyap)
        
    for cly, ax in zip(range(Nlyap),axs.flat):    
        
        label = labels[cly]
        ltp = lntypes[cly]
        if system=='lorenz96' or system=='cdv':
            ltp_lstm = 'r--'
        else: 
            ltp_lstm = lntypes_esn[cly]
        ax.tick_params(axis='y',pad=0.2)
        ax.set_title(titles[cly])
        ax.grid(True,c='lightgray',linestyle='--', linewidth=0.5)
        ax.set_xlabel(label,fontsize=fs)
        
        #PLOT TARGET
        ax.plot(PDF_x_targ[:,cly], PDF_y_targ[:,cly],'k-',lw=lw,label='Target');
            
        
        #PLOT LSTM
        ax.plot(PDF_x_lstm[:,cly], PDF_y_lstm[:,cly],ltp_lstm,lw=lw-1,label='LSTM');
            
        #set number of ticks
        # ax.locator_params(tight=True, nbins=4)
        
        wasser_dist_theta[cly] = wasserstein_distance(PDF_y_lstm[:,cly], PDF_y_targ[:,cly])
        print('Wasserstein distance x', cly, ': ', wasser_dist_theta[cly])
        ax.set_yscale('log')
        ax.set_xscale('log')

    fig.text(ylabelpos, 0.5, 'PDF',  fontsize=fs+2, ha='center', va='center', rotation='vertical')

    plt.savefig(img_filepath/f'clv_pdf.png', dpi=100, facecolor="w", bbox_inches="tight")
    plt.close()
    print('average Wasserstein distance:', np.average(wasser_dist_theta))
