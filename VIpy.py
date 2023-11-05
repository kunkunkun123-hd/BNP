import numpy as np
import time

def sampleELBOgrad(log_emp_dist, variationalDist, nSamples, varDistParams):
         
    dim = len(varDistParams['mu'])
    if variationalDist=='diagonalGauss':
        d_mu_mean = 0
        d_sigma_mean = 0
        d_muSq_mean = 0
        d_sigmaSq_mean = 0
        
        for i in range(1,nSamples+1):
            sample = np.random.normal(0, 1, dim)
            
            variationalSample = varDistParams['mu'] + varDistParams['sigma']*sample
            
            
            
            _, d_log_empirical,_ = log_emp_dist(variationalSample)

            d_log_empirical = d_log_empirical.T
            dle=np.zeros(d_log_empirical.shape)
            for k in range(d_log_empirical.shape[1]):
                dle[0,k]=d_log_empirical[0,k]
            
            d_mu_mean = (1./i)*((i - 1)*d_mu_mean + dle)
                         
            d_sigma_mean = (1./i)*((i - 1)*d_sigma_mean +\
                                   (dle*sample.reshape((1,-1)) + 1./varDistParams['sigma']))
            
            
            d_muSq_mean = (1./i)*((i - 1)*d_muSq_mean + dle**2)
            
            d_sigmaSq_mean = (1./i)*((i - 1)*d_sigmaSq_mean +\
                            (dle*sample.reshape((1,-1)) + 1./varDistParams['sigma'])**2)
        
        d_logSigma_Minus2mean = -.5*(d_sigma_mean*varDistParams['sigma'])

        ELBOgrad = np.hstack((d_mu_mean,d_logSigma_Minus2mean))

        d_muErr = np.sqrt(np.abs(d_muSq_mean - d_mu_mean**2))/np.sqrt(nSamples)

        
        dsE=.25*(varDistParams['sigma']**2)*d_sigmaSq_mean\
                             - d_logSigma_Minus2mean**2
        
        
        d_sigmaErr=np.sqrt(np.abs(dsE))/np.sqrt(nSamples)*np.sign(dsE)

        
        ELBOgradErr = np.hstack((d_muErr,d_sigmaErr))

    else:
         ValueError('Unknown variational distribution')
    return ELBOgrad, ELBOgradErr

def efficientStochOpt(x, log_emp_dist, variationalDist, stepWidth, dim, maxCompTime=10):
    
    debug = False   
    updateRule = 'amsgrad'
    # beta1 = .7                    
    # beta2 = .8                     
    beta1 = .9
    beta2 = .9999995
    epsilon = 1e-6                  

    stepOffset = 200000               
    maxIterations = np.Inf
    nSamplesStart = 1                  
    nSamplesEnd = 1
    nIncr = (nSamplesEnd - nSamplesStart)/maxCompTime
    nSamples = nSamplesStart

    converged = False
    steps = 0
    stepWidth_stepOffset = stepWidth*stepOffset
    if variationalDist == 'diagonalGauss':
        varDistParams = {}
        varDistParams['mu'] = x[0:dim]
        varDistParams['sigma'] = np.exp(-0.5*x[dim:])
    else:
        raise ValueError('Unknown variational distribution')
    cmpt=time.time()
    while converged==False:
        gradient,_=sampleELBOgrad(log_emp_dist, variationalDist, nSamples, varDistParams)
                
        if updateRule=='amsgrad':
            if steps==0:
                momentum = 1e-6*gradient
                uncenteredXVariance = gradient**2
                uncenteredXVariance_max = uncenteredXVariance
            else:
                momentum = beta1*momentum + (1 - beta1)*gradient
            uncenteredXVariance = beta2*uncenteredXVariance+ (1 - beta2)*gradient**2
            uncenteredXVariance_max[uncenteredXVariance_max<uncenteredXVariance]=uncenteredXVariance[uncenteredXVariance_max < uncenteredXVariance]
        
            
            x = x + (stepWidth_stepOffset/(stepOffset + steps))*\
                (1./(np.sqrt(uncenteredXVariance_max) + epsilon))*momentum

        else:
            raise ValueError('Unknown variational distribution')
        
        steps=steps+1

        if variationalDist == 'diagonalGauss':
            x=np.reshape(x,(-1,))
            varDistParams['mu'] = x[0:dim]
            varDistParams['sigma'] = np.exp(-0.5*x[dim:])
        else:
            raise ValueError('Unknown variational distribution')

        compTime=time.time()-cmpt
        nSamples = int(np.ceil(nIncr*compTime + nSamplesStart))
        if steps>maxIterations:
            converged=True
            print('Converged because max number of iterations exceeded')
        elif compTime>maxCompTime*5:
            converged=True
            print('Converged because max computation time exceeded')
    
  
    if variationalDist=='diagonalGauss':
       varDistParams['XSqMean'] = varDistParams['sigma']**2 + varDistParams['mu']**2
       
    else:
        raise ValueError('Unknown variational distribution') 
    
    return varDistParams, x  

def mcInference(functionHandle, variationalDist, varDistParams):
    
    inferenceSamples = 2000
    if variationalDist=='diagonalGauss':
        
        samples = np.random.multivariate_normal(varDistParams['mu'], np.diag(varDistParams['sigma']), inferenceSamples)
        E = 0
        E2 = 0
        E3 = 0
        
        for i in range(1,inferenceSamples+1):
            p_cf_exp, Tc, TcTcT = functionHandle(samples[i-1, :])
            E2 = (1./i)*((i - 1)*E2 + Tc)
            E3 = (1./i)*((i - 1)*E3 + TcTcT)
            E = (1./i)*((i - 1)*E + p_cf_exp)  
    else:
        raise ValueError('Unknown variational distribution')
    
    return E, E2, E3  
        

    

        