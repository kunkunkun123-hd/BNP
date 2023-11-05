from scipy.sparse import spdiags
from ModelRom import StokesROM
from ModelData import StokesData
from Params import ModelParams
import numpy as np
import os
import time
from Rompy import *
from Mythreadpy import MyThread
from VIpy import *
import shutil
import scipy.io as scio

path=os.path.dirname(os.path.abspath(__file__))
nTrain = 16
nStart = 0

samples = list(range(nStart, nStart + nTrain))
loadParams = False

rom = StokesROM()
rom.trainingData = StokesData(samples,path)
rom.trainingData.readData('px')
rom.trainingData.countVertices()

if loadParams:
    """
    print('loading modelParams...')
    """
else:
    rom.modelParams = ModelParams(rom.trainingData.u_bc, rom.trainingData.p_bc)
    rom.modelParams.initialize(nTrain)

    if any(rom.modelParams.interpolationMode):
        rom.trainingData.interpolate(rom.modelParams)   
        rom.modelParams.fineScaleInterp(rom.trainingData.X_interp)
        interp = True
    else:
        rom.modelParams.fineScaleInterp(rom.trainingData.X)
        interp = False

rom.modelParams.splitRFcells([])

if os.path.exists(path+'/state/'):
    shutil.rmtree(path+'/state/')  
os.mkdir(path+'/state/')  
if os.path.exists(path+'/htstate/'):
    shutil.rmtree(path+'/htstate/')   
os.mkdir(path+'/htstate/')  
if os.path.exists(path+'/finalstate/'):
    shutil.rmtree(path+'/finalstate/')  
os.mkdir(path+'/finalstate/') 
if not os.path.exists(path+'/savedata/'):
    os.mkdir(path+'/savedata/')

rom.trainingData.shiftData(interp, 'p') 
rom.trainingData.vtx2Cell(rom.modelParams) 

sw0_mu = 6e-4
sw0_sigma = 6e-5
sw_decay = .995 
VI_t = list(60*np.ones(1))+list(20*np.ones(10))+[10]
VI_t = [int(i) for i in VI_t]

split_schedule = []
nSplits = len(split_schedule)

tic_tot=time.time()
for split_iter in range(1,nSplits+2):
    rom.trainingData.designMatrix = [None] * nTrain
    rom.trainingData.evaluateFeatures(rom.modelParams.gridRF,path) 
    if rom.modelParams.normalization=='rescale':
        rom.trainingData.rescaleDesignMatrix(path+'/data/rescaling.mat')
    
    if rom.modelParams.mode=='local':
        rom.trainingData.shapeToLocalDesignMat()
    
    if len(rom.modelParams.theta_c)==0:
        print('Initializing theta_c...')
        rom.modelParams.theta_c =0*np.ones(((rom.trainingData.designMatrix[0].shape)[1], 1))
        print('...done.')

    nRFc = rom.modelParams.gridRF.nCells
    sw = (sw_decay**rom.modelParams.EM_iter)*np.concatenate((sw0_mu*np.ones((1, nRFc)), sw0_sigma*np.ones((1, nRFc))), axis=1)
    sw_min = 3e-1*np.concatenate((sw0_mu*np.ones((1, nRFc)), sw0_sigma*np.ones((1, nRFc))), axis=1)
    sw[sw < sw_min] = sw_min[sw < sw_min] 

    if len(rom.trainingData.a_x_m)==0:
        coarseMesh = rom.modelParams.coarseMesh 
        coarseMesh = coarseMesh.shrink() 
    else:
        '''vacancy'''

    converged=False
    pend = 0
    rom.modelParams.EM_iter_split = 0
    rom.modelParams.epoch = 0

    while converged==False:
        lg_q=[None]*nTrain
        for n in range(1,nTrain+1):
            P_n_minus_mu = rom.trainingData.P[n-1]
            if any(rom.modelParams.interpolationMode):
                W_cf_n = rom.modelParams.W_cf[0] 
                S_n = rom.modelParams.sigma_cf['s0'] 
            else:
                """vacancy"""

            S_cf_n = {}
            S_cf_n['sumLogS'] = np.sum(np.log(S_n))  
            S_cf_n['Sinv_vec'] = (1.0 / S_n).reshape((-1,))  
            Sinv = spdiags(S_cf_n['Sinv_vec'], 0, len(S_n), len(S_n)).tocsc()  
            S_cf_n['WTSinv'] = (Sinv@W_cf_n).T  

            tc = {}
            tc['theta'] = rom.modelParams.theta_c  
            tc['Sigma'] = rom.modelParams.Sigma_c
            if rom.modelParams.epoch>0:
                tc['Sigma']=tc['Sigma'].todense()  
            tc['SigmaInv'] = np.linalg.inv(tc['Sigma'])  

            Phi_n = rom.trainingData.designMatrix[n-1] 
            rf2fem = rom.modelParams.rf2fem  
            transType = rom.modelParams.diffTransform 
            transLimits = rom.modelParams.diffLimits 
            if len(rom.trainingData.a_x_m)==0:
                cm = coarseMesh
            else:
                """vacancy"""

            lg_q[n-1] = lambda Xi,P_n_minus_mu=P_n_minus_mu,Phi_n=Phi_n: log_q_n(Xi, P_n_minus_mu, W_cf_n, S_cf_n, tc, Phi_n, cm, transType, transLimits, rf2fem, True)
        
        varDistParamsVec = rom.modelParams.varDistParamsVec 
        
        if(rom.modelParams.epoch > 0 and loadParams==False):
            sw = sw_decay*sw
            sw[sw < sw_min] = sw_min[sw < sw_min]
        else:
            loadParams = False  

        pstart = 1
        pend = nTrain
        rom.modelParams.epoch = rom.modelParams.epoch + 1
               
        print('Variational Inference...')

        if (rom.modelParams.epoch) <= len(VI_t):
            t = VI_t[rom.modelParams.epoch]
        else:
            t = VI_t[-1]
        tic=time.time()
        threads = []
        for n in range(pstart,pend+1):
            thread = MyThread(efficientStochOpt,(varDistParamsVec[n-1], lg_q[n-1], 'diagonalGauss', sw, nRFc, t))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        varDistParams=[]

        jwl=pstart
        for thread in threads:
            mh,lxq=thread.get_result()
            varDistParams.append(mh)
            varDistParamsVec[jwl-1]=lxq
            jwl=jwl+1

        VI_time=time.time()-tic
        print('VI_time=',VI_time)
        rom.modelParams.varDistParamsVec = varDistParamsVec
        print('... VI done.')

        print('E-step...')
        tic=time.time()
        for n in range(pstart,pend+1):
            if n==pstart:
                rom.modelParams.variational_mu=[None]*pend
                rom.modelParams.variational_sigma=[None]*pend
                XMean=np.zeros((varDistParams[n-1]['mu'].shape[0],pend))
                XSqMean=np.zeros((varDistParams[n-1]['XSqMean'].shape[0],pend))
            rom.modelParams.variational_mu[n-1]=varDistParams[n-1]['mu']
            rom.modelParams.variational_sigma[n-1]=varDistParams[n-1]['sigma']
            XMean[:,n-1]=varDistParams[n-1]['mu']
            XSqMean[:,n-1]=varDistParams[n-1]['XSqMean']
            P_n_minus_mu = rom.trainingData.P[n-1]
            if any(rom.modelParams.interpolationMode):
                W_cf_n = rom.modelParams.W_cf[0] 
            else:
                """"""
            if len(rom.trainingData.a_x_m)==0:
                cm=coarseMesh
            else:
                """"""

            p_cf_expHandle_n = lambda X,P_n_minus_mu=P_n_minus_mu: sqMisfit(X, transType, transLimits,cm, P_n_minus_mu, W_cf_n, rf2fem)

            p_cf_exp,_,_ =mcInference(p_cf_expHandle_n,'diagonalGauss', varDistParams[n-1]) 
            if n==1:
                sqDist=[None]*pend
            sqDist[n-1]=p_cf_exp

        print('...E-step done')
        E_step_time = time.time()-tic
        print('E_step_time=',E_step_time)

        tic=time.time()
        print('M-step...')
        rom.M_step(XMean, XSqMean, sqDist)
        print('...M-step done.')
        M_step_time = time.time()-tic
        print('M_step_time=',M_step_time)

        rom.modelParams.compute_elbo(nTrain, XMean, XSqMean,rom.trainingData.X_interp[0])
        elbo = rom.modelParams.elbo
        print('elbo=',elbo)

        Lambda_eff1_mode = conductivityBackTransform(rom.trainingData.designMatrix[0]*rom.modelParams.theta_c,transType, transLimits) 
        print('Lambda_eff1_mode=\n',Lambda_eff1_mode)

        Lambda_c=rom.trainingData.designMatrix[0]@rom.modelParams.theta_c
        print("Lambda_c=\n",Lambda_c)

        rom.modelParams.printCurrentParams()
        plt = True

        if plt==True:
            print('Plotting...')
            t_plt=time.time()

            rom.modelParams.plot_params(path)

            rom.plotCurrentState(0, transType, transLimits,path,rom.modelParams.epoch)
            
            rom.modelParams.plotElbo(path)
            rom.modelParams.plotCellScores(path)
            t_plt = time.time()-t_plt
            print('...plotting done. Plotting time:',t_plt)
            
        
        rom.modelParams.write2file(path,'thetaPriorHyperparam')
        rom.modelParams.write2file(path,'theta_c')
        rom.modelParams.write2file(path,'sigma_c')
        rom.modelParams.write2file(path,'elbo')
        rom.modelParams.write2file(path,'cell_score')
        rom.modelParams.write2file(path,'cell_score_full')
        rom.modelParams.write2file(path,'sigma_cf_score')
        rom.modelParams.write2file(path,'inv_sigma_cf_score')

        if rom.modelParams.epoch > rom.modelParams.max_EM_epochs:
            converged = True

        scio.savemat(path+'/savedata/XMean.mat', {'XMean':XMean})
        scio.savemat(path+'/savedata/XSqMean.mat', {'XSqMean':XSqMean})

        print('epoch=',rom.modelParams.epoch)

      



