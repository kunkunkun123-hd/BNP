import numpy as np
import warnings
import scipy
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as scio
from FEMpy import *
from Rompy import *

class StokesROM:

    def __init__(self):
        
        self.trainingData = []
        self.modelParams = []
        self.pCurrState = []

    def update_a(self):
        if self.modelParams.prior_theta_c=='sharedVRVM':
            nElc = (self.trainingData.designMatrix[0]).shape[0]
            self.modelParams.a = self.modelParams.VRVM_a + .5*nElc
        elif self.modelParams.prior_theta_c=='adaptiveGaussian':
            dim_theta = len(self.modelParams.theta_c)
            self.modelParams.a = self.modelParams.VRVM_a + .5*dim_theta
        else:
            self.modelParams.a = self.modelParams.VRVM_a + .5
    
    def update_c(self):
        self.modelParams.c =self.modelParams.VRVM_c + .5*self.trainingData.nSamples
    
    def update_e(self):
        self.modelParams.e =self.modelParams.VRVM_e + .5*self.trainingData.nSamples
    
    def update_f(self, sqDist_p_cf):
        Ncells_gridS = len(self.modelParams.fineGridX)*len(self.modelParams.fineGridY)
        if any(self.modelParams.interpolationMode):
            sqDistSum = 0
            for n in range(1,len(self.trainingData.samples)+1):
                sqDistSum = sqDistSum + sqDist_p_cf[n-1]
        else:
            """"""
        self.modelParams.f = self.modelParams.VRVM_f + .5*sqDistSum 

    def initialize_gamma(self):
        dim_theta = len(self.modelParams.theta_c)
        if len(self.modelParams.gamma) != dim_theta:
            warnings.warn('resizing theta precision parameter gamma',UserWarning)
            self.modelParams.gamma = 1e0*np.ones((dim_theta, 1))

    def initialize_Sigma_theta_c(self):
        if len(self.modelParams.Sigma_theta_c)==0:
            dim_theta = len(self.modelParams.theta_c)
            nElc = self.trainingData.designMatrix[0].shape[0]
            if self.modelParams.diag_theta_c:
                self.modelParams.Sigma_theta_c = 1./self.modelParams.gamma
            else:
                """"""

    def get_designMatrixSqSum(self):
        try:
            if len(self.trainingData.designMatrixSqSum)==0:
                dim_theta = len(self.modelParams.theta_c)
                nElc = self.trainingData.designMatrix[0].shape[0]
                if self.modelParams.diag_theta_c:
                    self.trainingData.designMatrixSqSum = 0
                    for n in range(1,self.trainingData.nSamples+1):
                        dsq=self.trainingData.designMatrix[n-1]*np.eye(self.trainingData.designMatrix[n-1].shape[1])
                        self.trainingData.designMatrixSqSum = self.trainingData.designMatrixSqSum + dsq**2
                    self.trainingData.designMatrixSqSum = scipy.sparse.csr_matrix(self.trainingData.designMatrixSqSum.T)
                else:
                    """"""
        except:
            """"""

    def compute_theta_cSq(self):
        
        if self.modelParams.Sigma_theta_c.shape[0]==1 or self.modelParams.Sigma_theta_c.shape[1]==1:
            self.modelParams.theta_cSq = self.modelParams.theta_c**2 +self.modelParams.Sigma_theta_c
        else:
            self.modelParams.theta_cSq = self.modelParams.theta_c**2 +np.diag(self.modelParams.Sigma_theta_c)

    def update_b(self):
        dim_theta = len(self.modelParams.theta_c)
        
        nElc = self.trainingData.designMatrix[0].shape[0]  
        dim_gamma = int(dim_theta/nElc) 
        if self.modelParams.prior_theta_c=='sharedVRVM':
            thetaSq_expect_sum = np.sum(np.reshape(self.modelParams.theta_cSq,(dim_gamma, nElc),order='F'), axis=1).reshape((-1,1)) 
            b = self.modelParams.VRVM_b + .5*thetaSq_expect_sum
            self.modelParams.b = np.tile(b, (nElc, 1)) 
        else:
            """"""
    
    def update_d(self, XMean, XSqMean):
            
            d = self.modelParams.VRVM_d + .5*np.sum(XSqMean, axis=1)
            PhiThetaMean_n_sq_sum = 0
            for n in range(1,self.trainingData.nSamples+1):
                PhiThetaMean_n = (self.trainingData.designMatrix[n-1]@self.modelParams.theta_c).reshape((-1,))
                PhiThetaMean_n_sq_sum = PhiThetaMean_n_sq_sum + PhiThetaMean_n**2
                d = d - XMean[:, n-1]*PhiThetaMean_n
            
            if self.modelParams.diag_theta_c:
                self.modelParams.d = d + .5*(PhiThetaMean_n_sq_sum + \
                    (self.trainingData.designMatrixSqSum.T@self.modelParams.Sigma_theta_c).reshape(-1,))
            else:
                """"""
            self.modelParams.d=self.modelParams.d.reshape((-1,1))
    
    def compute_Sigma_theta_c(self, Phi_full, I, opts):
            #short hand notation
            nElc = self.trainingData.designMatrix[0].shape[0] 
            dim_theta = np.prod(self.modelParams.theta_c.shape) 
            nFeatures = int(dim_theta/nElc) 
            tau_c = (1/np.diag(self.modelParams.Sigma_c@np.eye(self.modelParams.Sigma_c.shape[0]))).reshape((-1,1))
            
            if self.modelParams.diag_theta_c:
                tau_theta = self.trainingData.designMatrixSqSum@tau_c +self.modelParams.gamma
                
                self.modelParams.Sigma_theta_c = 1./tau_theta
            else:
                """"""
    
    def compute_theta_c(self, XMean, Phi_full):
        nElc =self.trainingData.designMatrix[0].shape[0]
        tau_c = (1/np.diag(self.modelParams.Sigma_c@np.eye(self.modelParams.Sigma_c.shape[0])))
        sumPhiTau_cXMean = 0
        for n in range(1,self.trainingData.nSamples+1):
            sumPhiTau_cXMean = sumPhiTau_cXMean + self.trainingData.designMatrix[n-1].T@np.diag(tau_c)@(XMean[:, n-1]).reshape((-1,1))        
        
        if self.modelParams.diag_theta_c:
            tau_c_long=scipy.sparse.csr_matrix((np.tile(tau_c, (1, self.trainingData.nSamples)).reshape((-1,)),(np.arange(int(nElc*self.trainingData.nSamples)),np.arange(int(nElc*self.trainingData.nSamples)))))
            
            dim_theta = np.prod(self.modelParams.theta_c.shape)
            nFeatures = dim_theta/nElc
            A = (Phi_full.T@(tau_c_long@Phi_full))@np.diag((self.modelParams.Sigma_theta_c).reshape((-1,))) 
            A_diag = np.diag(A).reshape((-1,1)) 
            sumPhiT_tau_cXMean_Sigma_plus_A_diag_theta = \
                self.modelParams.Sigma_theta_c*sumPhiTau_cXMean + \
                A_diag*self.modelParams.theta_c 
            tc = self.modelParams.theta_c.T 

            nCycles = 1
            for c in range(1,nCycles+1):
                for j in range(1,dim_theta+1):
                    term2 = tc.reshape((-1,))@A[:, j-1]
                    tc[0,j-1] =sumPhiT_tau_cXMean_Sigma_plus_A_diag_theta[j-1,0] - term2            
            self.modelParams.theta_c = tc.T
        else:
            """"""
                                  
    def M_step(self, XMean, XSqMean, sqDist_p_cf):
            
        if self.modelParams.prior_theta_c=='VRVM' or \
                self.modelParams.prior_theta_c=='sharedVRVM' or \
                self.modelParams.prior_theta_c=='adaptiveGaussian':
            dim_theta = len(self.modelParams.theta_c) 
            nElc = (self.trainingData.designMatrix[0]).shape[0] 
            nFeatures = int(dim_theta/nElc) 

            
            self.update_a()
            self.update_c()
            self.update_e()
            self.update_f(sqDist_p_cf)
            tau_cf = (self.modelParams.e)/(self.modelParams.f)

            
            self.initialize_gamma() 
            self.initialize_Sigma_theta_c()

            
            if self.modelParams.diag_theta_c:
                    I = []     
                    opts = []
            else:
                """"""
            
            for mh in range(len(self.trainingData.designMatrix)):
                if mh==0:
                    Phi_full=self.trainingData.designMatrix[mh].todense()
                else:
                    Phi_full = np.concatenate((Phi_full,self.trainingData.designMatrix[mh].todense()), axis=0)
            Phi_full=scipy.sparse.csr_matrix(Phi_full) 
            self.get_designMatrixSqSum()

            
            if self.modelParams.epoch + 1 <=len(self.modelParams.VRVM_time):
                t_max = self.modelParams.VRVM_time[self.modelParams.epoch]
            else:
                t_max = self.modelParams.VRVM_time[-1]
            
            if self.modelParams.epoch + 1 <=len(self.modelParams.VRVM_iter):
                iter_max = self.modelParams.VRVM_iter[self.modelParams.epoch]
            else:
                iter_max = self.modelParams.VRVM_iter[-1]
            
            
            converged = False
            i = 0
            cmpt = time.time()
            while converged==False:
                self.compute_theta_cSq() 
          
                self.update_b()
                
                
                self.modelParams.gamma =(self.modelParams.a)/(self.modelParams.b) 
                            
                self.update_d(XMean, XSqMean) 
                                           
                
                self.modelParams.Sigma_c = scipy.sparse.csr_matrix(np.diag(((self.modelParams.d)/(self.modelParams.c)).reshape((-1,))))
              
                
                self.compute_Sigma_theta_c(Phi_full, I, opts)

                self.compute_theta_c(XMean, Phi_full)
                
                i = i + 1
                t = time.time()-cmpt
                if (t > t_max*5 or i >= iter_max):
                    converged = True
            
            
            self.modelParams.sigma_cf['s0'] = 1./tau_cf  
            mean_s0 = np.mean(self.modelParams.sigma_cf['s0'])
            print('mean_s0=',mean_s0)

        self.modelParams.EM_iter = self.modelParams.EM_iter + 1
        self.modelParams.EM_iter_split = self.modelParams.EM_iter_split + 1    

    def plotCurrentState(self, dataOffset, transType, transLimits,path,epoch):

        fig=plt.figure()

        for i in range(1,4+1):
            Lambda_eff_mode = conductivityBackTransform(\
                self.trainingData.designMatrix[i - 1 + dataOffset]@self.modelParams.theta_c, transType, transLimits) 

            Lambda_eff_mode = self.modelParams.rf2fem@Lambda_eff_mode           
            
            sb1 = plt.subplot(4, 2, 1 + (i - 1)*2)
            im=sb1.imshow(np.reshape(Lambda_eff_mode,\
                (self.modelParams.coarseMesh.nElX,\
                self.modelParams.coarseMesh.nElY),order='F').T,origin='lower')
            
            plt.colorbar(im)
            sb1.set_xticks([])
            sb1.set_yticks([])
            sb1.grid(False)
            s11=np.reshape(Lambda_eff_mode,\
                (self.modelParams.coarseMesh.nElX,\
                self.modelParams.coarseMesh.nElY),order='F').T
            scio.savemat(path+'/state/Lambda_eff_mode_'+str(i-1)+'_'+'epoch='+str(epoch)+'.mat',{'s1':s11})
            
            if len(self.trainingData.cells)==0:
                    self.trainingData.readData('c')
            
            s21=self.trainingData.cells[i-1 + dataOffset]
            s22=self.trainingData.X[i-1 + dataOffset][:, 0]
            s23=self.trainingData.X[i-1 + dataOffset][:, 1]
            s24=np.zeros(self.trainingData.X[i-1 + dataOffset].shape[0])
            scio.savemat(path+'/state/micro_'+str(i-1)+'_'+'epoch='+str(epoch)+'.mat',{'s1':s21,'s2':s22,'s3':s23,'s4':s24})
    
            
            sb3 = fig.add_subplot(4, 2, 2 + (i - 1)*2, projection='3d')   
            if len(self.trainingData.a_x_m)==0:
                
                coarseMesh = self.modelParams.coarseMesh
                coarseMesh = coarseMesh.shrink()
            else:
                """"""
            isotropicDiffusivity = True
            if isotropicDiffusivity:
                coarseFEMout =heat2d(coarseMesh, Lambda_eff_mode)
            else:
                """"""
            Tc = coarseFEMout['u'] 
            nx = np.prod(self.modelParams.fineGridX.shape) + 1
            ny = np.prod(self.modelParams.fineGridY.shape) + 1
            XX = np.reshape(self.trainingData.X_interp[0][:, 0], (nx, ny),order='F')
            YY = np.reshape(self.trainingData.X_interp[0][:, 1], (nx, ny),order='F')
            
            if any(self.modelParams.interpolationMode):
                reconstruction = np.reshape(self.modelParams.W_cf[0]@Tc,(nx,ny),order='F')

                sb3.plot_surface(XX,YY,reconstruction,cmap=plt.cm.Blues)
            else:
                """"""
            if any(self.modelParams.interpolationMode):
                    P = np.reshape(self.trainingData.P[i-1 + dataOffset], (nx, ny),order='F')
                    sb3.plot_surface(XX, YY, P,cmap='magma')
            else:
                """"""
            scio.savemat(path+'/state/p_'+str(i-1)+'_'+'epoch='+str(epoch)+'.mat',{'X':XX,'Y':YY,'Pr':reconstruction,'P':P})
            sb3.view_init(azim=-127.5, elev=30)

        plt.savefig(path+'/state/currentstate_epoch='+str(epoch)+'.eps')


        
    
