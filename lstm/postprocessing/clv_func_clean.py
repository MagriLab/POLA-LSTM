import numpy as np
import math 
import h5py
import itertools
import scipy

def normalize(M):
    ''' Normalizes columns of M individually '''
    nM = np.zeros(M.shape) # normalized matrix
    nV = np.zeros(np.shape(M)[1]) # norms of columns

    for i in range(M.shape[1]):
        nV[i] = scipy.linalg.norm(M[:,i])
        nM[:,i] = M[:,i] / nV[i]

    return nM, nV

def timeseriesdot(x,y, multype):
    tsdot = np.einsum(multype,x,y.T) #Einstein summation. Index i is time.
    return tsdot

def CLV_consistency(eom, params, state, clv, flag, dt):

    if flag==True:
        #compute dq/dt
        time_state = np.shape(state)[0]
        print('CLV_consistency: time_state',time_state)
        v_dqdt = np.array([ eom(state[t,:], params) for t in range(time_state)])
    else:
        #compute delta h(i+1) - h(i)/dt
        v_dqdt = (state[1:]-state[:-1])/dt

    #normalize v_dqdt
    v_dqdt = vec_normalize(v_dqdt,0)

    time_v_dqdt = np.shape(v_dqdt)[0]
    time_clv = np.shape(clv)[-1]
    timetot = min(time_v_dqdt, time_clv)

    #Calculate the dot product of the neutral CLV and dq/dt
    costhetas_dqdt = timeseriesdot(clv[:,:timetot],v_dqdt[:timetot].T,'ij,ji->j')

    #Extract the angle from costheta
    thetas = 180. * np.arccos(costhetas_dqdt) / math.pi
    print('thetas shape',np.shape(thetas))

    return thetas

def vec_normalize(vec,timeaxis):
    #normalize a vector within its timeseries
    timetot = np.shape(vec)[timeaxis]
    for t in range(timetot):
        vec[t,:] = vec[t,:] / np.linalg.norm(vec[t,:])
    return vec

def subspace_angles(clv, timeaxis, index):
    timetot = np.shape(clv)[timeaxis]

    #calculate angles between subspaces
    thetas = np.zeros((timetot,3))

    #Nv_un and clvs of the unstable expanding subspace
    Nv_un = index[0]
    CLV_un = clv[:,0:Nv_un,:].copy()
    print('CLV_un shape', CLV_un.shape)
    pos_clvs = Nv_un

    #Nv_nu and clvs of the neutral subspace
    Nv_nu = index[1]
    CLV_nu = clv[:,Nv_un:Nv_un+Nv_nu,:].copy()
    print('CLV_nu shape', CLV_nu.shape)
    neut_clvs = Nv_nu

    #clvs of the stable decaying subspace
    CLV_st = clv[:,Nv_un+Nv_nu:,:].copy()
    print('CLV_st shape', CLV_st.shape)
    neg_clvs = np.shape(CLV_st)[1]

    for t in range(timetot):

        thetas[t,0] = np.rad2deg(scipy.linalg.subspace_angles(CLV_un[:,:,t],CLV_nu[:,:,t]))[-1]
        thetas[t,1] = np.rad2deg(scipy.linalg.subspace_angles(CLV_un[:,:,t],CLV_st[:,:,t]))[-1]
        thetas[t,2] = np.rad2deg(scipy.linalg.subspace_angles(CLV_nu[:,:,t],CLV_st[:,:,t]))[-1]

    return thetas

def CLV_angles(clv, NLy):
    #calculate angles between CLVs
    thetas_num = int(np.math.factorial(NLy) / (np.math.factorial(2) * np.math.factorial(NLy-2)))
    costhetas = np.zeros((clv[:,0,:].shape[1],thetas_num))
    count = 0
    for subset in itertools.combinations(np.arange(NLy), 2):
        index1 = subset[0]
        index2 = subset[1]
        #For principal angles take the absolute of the dot product
        costhetas[:,count] = np.absolute(timeseriesdot(clv[:,index1,:],clv[:,index2,:],'ij,ji->j'))
        count+=1
    thetas = 180. * np.arccos(costhetas) / math.pi

    return thetas

def CLV_calculation(QQ, RR, NLy, n_cells_x2, dt, subspace_LEs_indeces, fname=None, system=None):
    """
    Calculates the Covariant Lyapunov Vectors (CLVs) using the Ginelli et al, PRL 2007 method.

    Args:
    - QQ (numpy.ndarray): matrix containing the timeseries of Gram-Schmidt vectors (shape: (n_cells_x2,NLy,tly))
    - RR (numpy.ndarray): matrix containing the timeseries of upper-triangualar R  (shape: (NLy,NLy,tly))
    - NLy (int): number of Lyapunov exponents
    - n_cells_x2 (int): dimension of the hidden states
    - dt (float): integration time step
    - subspace_LEs_indeces (numpy.ndarray): indices of the Lyapunov exponents signs for positive and neutral. (shape: (2,))

    Returns:
    - nothing
    """
    tly = np.shape(QQ)[-1]
    su = int(tly / 10)
    sd = int(tly / 10)
    s  = su          # index of spinup time
    e  = tly+1 - sd  # index of spindown time
    tau = int(dt/dt)     #time for finite-time lyapunov exponents

    #Calculation of CLVs
    C = np.zeros((NLy,NLy,tly))  # coordinates of CLVs in local GS vector basis
    D = np.zeros((NLy,tly))  # diagonal matrix
    V = np.zeros((n_cells_x2,NLy,tly))  # coordinates of CLVs in physical space (each column is a vector)

    # FTCLE
    il  = np.zeros((NLy,tly+1)) #Finite-time lyapunov exponents along CLVs

    # initialise components to I
    C[:,:,-1] = np.eye(NLy)
    D[:,-1]   = np.ones(NLy)
    V[:,:,-1] = np.dot(np.real(QQ[:,:,-1]), C[:,:,-1])

    for i in reversed(range( tly-1 ) ):
        C[:,:,i], D[:,i]        = normalize(scipy.linalg.solve_triangular(np.real(RR[:,:,i]), C[:,:,i+1]))
        V[:,:,i]                = np.dot(np.real(QQ[:,:,i]), C[:,:,i])

    # FTCLE computations
    for j in 1+np.arange(s, e): #time loop
        il[:,j] = -(1./dt)*np.log(D[:,j])
        

    #normalize CLVs before measuring their angles.
    timetot = np.shape(V)[-1]

    for i in range(NLy):
        for t in range(timetot-1):
            V[:,i,t] = V[:,i,t] / np.linalg.norm(V[:,i,t])
            

    if system == 'lorenz96' or system == 'cdv':
        #Compute the subspace angles between the CLVs.
        # E.g for L96 at D=20: subspace_LEs_indeces = [6,1]
        # i.e. 6 positive LEs and 1 neutral
        thetas_clv = subspace_angles(V, -1, subspace_LEs_indeces)

        #these are angles between all pairs of the first num_first_clvs CLVs.
        num_first_clvs = 5
        thetas_clv_indiv = CLV_angles(V, num_first_clvs)
    else:
        thetas_clv = CLV_angles(V, NLy)

    '''Save the data to hdf5 file'''
    if fname is not None:
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('thetas_clv',       data=thetas_clv)
            hf.create_dataset('thetas_clv_indiv', data=thetas_clv_indiv)
            hf.create_dataset('FTCLE',            data=il)
            
    return thetas_clv, il, D