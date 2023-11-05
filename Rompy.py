import numpy as np
from FEMpy import *

def log_p_c(Xq, Phi, theta_c):
    
    
    mu = Phi @ theta_c['theta']  # mean 16*1
    
    mu=mu.reshape((-1,))
    log_p = -0.5 * np.log(np.linalg.det(theta_c['Sigma'])) - 0.5 * (Xq - mu) @ theta_c['SigmaInv'] @ (Xq - mu)   
    d_log_p = theta_c['SigmaInv'] @ (mu - Xq)
    data=0
    return log_p,d_log_p,data
    

def conductivityBackTransform(x, transType, transLimits):

    if transType=='log':
        conductivity = np.exp(x)
        if (conductivity < transLimits[0]).any():
            
            conductivity[conductivity < transLimits[0]] = transLimits[0]
        if (conductivity > transLimits[1]).any():
            
            conductivity[conductivity > transLimits[1]] = transLimits[1]
    else:
        raise ValueError('unknown conductivity transformation')

    if (~np.isfinite(conductivity)).any():
        print('Non-finite conductivity, setting it to 1e-3.')
        conductivity[~np.isfinite(conductivity)] = 1e-3
    return conductivity

def log_p_cf(Tf_n_minus_mu, coarseMesh,Xn,W_cf_n, S_cf_n, transType, transLimits, rf2fem, onlyGrad):
    

    conductivity = conductivityBackTransform(Xn, transType, transLimits)

    conductivity = rf2fem@conductivity

    isotropicDiffusivity = True
    if isotropicDiffusivity:
        FEMout = heat2d(coarseMesh, conductivity)
    else:
        """"""

    Tc = FEMout['u']
    
    Tf_n_minus_mu_minus_WTc = Tf_n_minus_mu - np.reshape(W_cf_n@Tc,-1)

    if onlyGrad:
        
        log_p = []
    else:
        log_p = -.5*(S_cf_n.sumLogS +(S_cf_n.Sinv_vec@(Tf_n_minus_mu_minus_WTc**2)))
        
        
    d_r = FEMgrad(FEMout['naturalTemperatures'], coarseMesh)

    if transType=='log':
        
        d_rx=np.diag(conductivity)@d_r
        """
        """
    else:
        raise ValueError('Unknown conductivity transformation?')
    adjoints = get_adjoints(FEMout['globalStiffness'],S_cf_n['WTSinv'], coarseMesh, Tf_n_minus_mu_minus_WTc)
    d_log_p = - d_rx@(adjoints).reshape((-1,1),order='F')
    
    d_log_p = rf2fem.T@d_log_p
    
    
    FDcheck = False
    return log_p, d_log_p, Tc


def log_q_n(Xn, Tf_n_minus_mu, W_cf_n, S_cf_n, theta_c, designMatrix, coarseMesh,
            transType, transLimits, rf2fem, onlyGrad):
    
    try:
        if Xn.shape[1] > 1:
            Xn = Xn.T   
    except:
        """"""
    lg_p_c, d_lg_p_c, _ = log_p_c(Xn, designMatrix, theta_c)
    lg_p_cf, d_lg_p_cf, Tc = log_p_cf(Tf_n_minus_mu, coarseMesh, Xn,
                                       W_cf_n, S_cf_n, transType, transLimits,
                                       rf2fem, onlyGrad)

    log_q = lg_p_cf + lg_p_c
    d_log_q = d_lg_p_c.reshape((-1,1),order='F') + d_lg_p_cf

    
    FDcheck = False
    return log_q, d_log_q, Tc

def sqMisfit(X, transType, transLimits, mesh, Tf_n_minus_mu, W_cf_n, rf2fem):
    
    X=X.reshape((1,-1))
    conductivity = conductivityBackTransform(rf2fem@X.T, transType, transLimits)
    isotropicDiffusivity = True
    if isotropicDiffusivity:
        FEMout = heat2d(mesh, conductivity)
    else:
        """
        """

    Tc = FEMout['u'].reshape((-1,1)) 

    p_cf_exp = (Tf_n_minus_mu.reshape((-1,1)) - W_cf_n@Tc)**2

    TcTcT = Tc@(Tc.T)
    return p_cf_exp, Tc, TcTcT