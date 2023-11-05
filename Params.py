import numpy as np
from Meshpy import RectangularMesh
from FEMpy import MeshFEM,shapeInterp
import warnings
import scipy.io as scio
import matplotlib.pyplot as plt
import os 

class ModelParams:

    def __init__(self,u_bc,p_bc):
        
        self.theta_c = []
        self.theta_cSq = []
        self.Sigma_c = []
        
        self.Sigma_theta_c = []
        
        self.coarseGridX = 1/16*np.ones(16)
        self.coarseGridY = 1/16*np.ones(16)
        
        self.gridRF = []
        self.splitted_cells = []

        
        self.sum_in_macrocell = []

        
        self.elbo = []
        self.cell_score = []
        self.cell_score_full = []   
        self.active_cells = []
        self.active_cells_S = []
        self.sigma_cf_score = []    
        self.inv_sigma_cf_score = []    

        
        self.W_cf = []
        self.sigma_cf = {}
        self.fineGridX = 1/128*np.ones(128)
        self.fineGridY = 1/128*np.ones(128)
        self.interpolationMode = 'linear'   
        self.smoothingParameter = []
        self.boundarySmoothingPixels = -1   

        
        self.rf2fem = []
        
        self.cell_dictionary = []

        
        self.diffTransform = 'log'
        self.diffLimits = [1e-10, 1e8]
        
        self.mode = 'local'  
        
        self.prior_theta_c = 'sharedVRVM'
        self.diag_theta_c = True     
        self.gamma = None   
        self.VRVM_a = np.finfo(float).eps
        self.VRVM_b = np.finfo(float).eps
        self.VRVM_c = np.finfo(float).eps
        self.VRVM_d = np.finfo(float).eps
        self.VRVM_e = np.finfo(float).eps
        self.VRVM_f = np.finfo(float).eps
        
        self.VRVM_iter = [np.inf]
        self.VRVM_time = np.append(np.concatenate((np.ones(10), 2*np.ones(10))),5)

        
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.e = []
        self.f = []

        
        self.varDistParamsVec = []
        self.variational_mu = []
        self.variational_sigma = []

        
        self.normalization = 'rescale'
        self.featureFunctionMean = []
        self.featureFunctionSqMean = []
        self.featureFunctionMin = []
        self.featureFunctionMax = []

        
        self.EM_iter = 0 
        self.EM_iter_split = 0 
        self.epoch = 0 
        self.max_EM_epochs = 1 

        
        self.computeElbo = True

        
        self.plt_params = True 
        self.pParams = {}
        self.pElbo = []
        self.pCellScores = []
        
        self.gridRF = RectangularMesh((1/4)*np.ones(4))
        self.cell_dictionary = np.arange(1,self.gridRF.nCells+1)
        
        self.coarseMesh = MeshFEM(self.coarseGridX, self.coarseGridY)
        self.coarseMesh.compute_grad = True
        self.coarseMesh_geometry = RectangularMesh(self.coarseGridX,self.coarseGridY)           
        
        
        u_x_temp = u_bc[0][4:].replace('x[1]', '*y')
        
        u_x_temp_le = u_x_temp.replace('x[0]', '0')
        u_x_temp_r = u_x_temp.replace('x[0]', '1')
        u_y_temp = u_bc[1][4:].replace('x[0]', '*x')
        u_y_temp_lo = u_y_temp.replace('x[1]', '0')
        u_y_temp_u = u_y_temp.replace('x[1]', '1')
            
        u_bc_handle = [None] * 4
        u_bc_handle[0] = lambda x: -1 * eval(u_y_temp_lo)
        u_bc_handle[1] = lambda y: eval(u_x_temp_r)
        u_bc_handle[2] = lambda x: eval(u_y_temp_u)
        u_bc_handle[3] = lambda y: -1 * eval(u_x_temp_le)

        p_bc_handle = lambda x: eval(p_bc)

        nX = len(self.coarseGridX)
        nY = len(self.coarseGridY)

        self.coarseMesh = self.coarseMesh.setBoundaries(list(range(2, 2*nX + 2*nY+1)), p_bc_handle, u_bc_handle)
        self.Elbox=[]
        self.Elboy=[]

    def initialize(self, nData):
        
        self.Sigma_c = 1e0 * np.eye(self.gridRF.nCells)
        
        nSX = len(self.fineGridX)
        nSY = len(self.fineGridY)
        
        if any(self.interpolationMode):
            nSX += 1
            nSY += 1
        
        self.sigma_cf['s0'] = np.ones(nSX * nSY)  
        
        
        if self.prior_theta_c == 'sharedVRVM':
            self.gamma = 1e-4 * np.ones_like(self.theta_c)
        else:
            raise ValueError('What prior model for theta_c?')
        
        self.variational_mu = -8.5 * np.ones(self.gridRF.nCells)
        self.variational_mu = [self.variational_mu] * nData
        
        self.variational_sigma = .25e-1 * np.ones(self.gridRF.nCells)
        self.variational_sigma = [self.variational_sigma] * nData
        
        varDistParamsVecInit = np.concatenate((self.variational_mu[0], -2 * np.log(self.variational_sigma[0])), axis=None)
        self.varDistParamsVec = np.tile(varDistParamsVecInit, (nData, 1)) 
    

    def fineScaleInterp(self, X):
                       
            nData = len(X)
            self.W_cf=[]
            for n in range(nData):               
                mh=shapeInterp()
                self.W_cf.append(mh.shapeInterp1(self.coarseMesh, X[n]))

    def splitRFcells(self, splt_cells):
         nElc=self.Sigma_c.shape[0]
         self.splitted_cells.append(splt_cells)
         
         self.rf2fem=self.gridRF.map2fine(self.coarseMesh_geometry)

    def set_summation_matrix(self, X):
        
        nx = np.prod(self.fineGridX.shape) + 1 #129
        ny = np.prod(self.fineGridY.shape) + 1
        self.sum_in_macrocell = np.zeros((self.gridRF.nCells, nx*ny))
        kk = 1
        for k in range(1,len(self.gridRF.cells)+1):
            try:
                self.sum_in_macrocell[kk-1, :] =\
                    (self.gridRF.cells[k-1].inside(X))
                kk = kk + 1
            except:
                print('self.gridRF.cells['+str(k-1)+'] is not valid')

    def compute_elbo(self, N, XMean, XSqMean, X_vtx):
        
        print('computing elbo... ')
        assert len(self.interpolationMode) !=0, 'Elbo only implemented with fixed dim(U_f)'
        
        N_dof = np.prod(self.fineGridX.shape)*np.prod(self.fineGridY.shape) 
        D_c = self.gridRF.nCells 
        aa = self.VRVM_a
        bb = self.VRVM_b
        cc = self.VRVM_c
        dd = self.VRVM_d
        ee = self.VRVM_e
        ff = self.VRVM_f
        D_theta_c = np.prod(self.theta_c.shape) 
        nFeatures = int(D_theta_c/D_c) 

        if self.prior_theta_c=='sharedVRVM':
            D_gamma = nFeatures  
        else:
            D_gamma = D_theta_c
        
        
        Sigma_lambda_c = XSqMean - XMean**2 
        
        sum_logdet_lambda_c = np.sum(np.log(Sigma_lambda_c)) 
        try:
            if self.diag_theta_c:
                logdet_Sigma_theta_ck = np.zeros(D_c)
                if self.mode=='local':
                    for k in range(1,D_c+1):
                        logdet_Sigma_theta_ck[k-1] = \
                            sum(np.log(self.Sigma_theta_c[\
                            ((k-1)*nFeatures):(k*nFeatures),0]))                   
                    logdet_Sigma_theta_c = np.sum(logdet_Sigma_theta_ck)
                else:
                    """"""
            else:
                """"""
        except:
            logdet_Sigma_theta_c = np.log(np.det(self.Sigma_theta_c))
            warnings.warn('resizing theta precision parameter gamma',UserWarning)

        self.elbo = -.5*N*N_dof*np.log(2*np.pi) +.5*sum_logdet_lambda_c + \
            .5*N*D_c + N_dof*(ee*np.log(ff) + np.math.lgamma(self.e) -\
            np.math.lgamma(ee)) - self.e*np.sum(np.log(self.f)) + D_c*(cc*np.log(dd) +\
            np.math.lgamma(self.c) - np.math.lgamma(cc)) -\
            self.c*np.sum(np.log(self.d)) + D_gamma*(aa*np.log(bb) +\
            np.math.lgamma(self.a) - np.math.lgamma(aa)) - \
            self.a*np.sum(np.log(self.b[:D_gamma])) + \
            .5*logdet_Sigma_theta_c + .5*D_theta_c

        
        self.set_summation_matrix(X_vtx)

        self.cell_score = .5*np.sum(np.log(Sigma_lambda_c), axis=1) - self.c*(np.log(self.d).reshape((-1,)))
        
        if self.prior_theta_c=='sharedVRVM':
            self.cell_score = self.cell_score + .5*logdet_Sigma_theta_ck 

        f_contribution = - self.e*np.log(self.f) 
        
        self.cell_score_full = self.cell_score.reshape((-1,1)) +self.sum_in_macrocell@f_contribution 

        
        self.sigma_cf_score = self.sum_in_macrocell@np.sqrt(self.sigma_cf['s0'])
        self.inv_sigma_cf_score =\
            self.sum_in_macrocell@np.sqrt(1./self.sigma_cf['s0']) 
    
    def printCurrentParams(self):
        
        if self.mode=='local':
            print('theta_c: row = feature, column = macro-cell:')
            curr_Sigma_c = np.diag(self.Sigma_c.todense())
            print('curr_Sigma_c=\n',curr_Sigma_c)
            if self.prior_theta_c=='sharedVRVM':
                curr_gamma = self.gamma[:int(np.prod(self.theta_c.shape)/self.gridRF.nCells),0] 
            else:
                curr_gamma = self.gamma
        else:
            """"""
        if self.prior_theta_c=='sharedVRVM' or \
                self.prior_theta_c=='VRVM':
            activeFeatures =np.concatenate(((np.where(curr_gamma < 20)[0]).reshape((-1,1)), (curr_gamma[curr_gamma < 20]).reshape((-1,1))),axis=1)
        
        print('activeFeatures=\n',activeFeatures)
    
    def plot_params(self,path):
        
        
        plt.figure()
               
        nSX = len(self.fineGridX)
        nSY = len(self.fineGridY)
        if any(self.interpolationMode):
            nSX += 1
            nSY += 1
        
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]])
        
        theta_c=self.theta_c.reshape((-1,))
        sb2 = plt.subplot(3, 2, 2)
        
        sb2.bar(np.arange(theta_c.shape[0]),theta_c)
        sb2.set_xlabel('component $i$')
        sb2.set_ylabel(r'$\theta_{c,i}$')

        sb3 = plt.subplot(3, 2, 3)
        
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]])
        Sigma_c=self.Sigma_c.todense()
        dsc=np.diag(Sigma_c)
        if self.EM_iter==1:
            self.pParams['p_sigma']=[None]*self.gridRF.nCells    
            for d in range(1,self.gridRF.nCells+1):
                self.pParams['p_sigma'][d-1]=[]  
        for d in range(1,self.gridRF.nCells+1):
            self.pParams['p_sigma'][d-1].append(dsc[d-1])
            sb3.plot(np.arange(1,self.EM_iter+1),self.pParams['p_sigma'][d-1],color=colors[(d % 6),:])            
        sb3.set_xlabel('iter')
        sb3.set_ylabel(r'$\sigma_c$')
        sb3.set_yscale('log')


        sb4 = plt.subplot(3, 2, 4)
         
        sigma_c_plot = np.sqrt(self.rf2fem@np.diag(Sigma_c))
        im=sb4.imshow(sigma_c_plot.reshape((self.coarseMesh.nElX, self.coarseMesh.nElY),order='F').T,origin='lower')
        sb4.set_title(r'$\sigma_c$')
        plt.colorbar(im)
        sb4.set_xticks([])
        sb4.set_yticks([])
        sb4.grid(False)

        
        sb5 = plt.subplot(3, 2, 5)
        if self.EM_iter==1:
            self.pParams['p_gamma']=[None]*int(np.prod(self.theta_c.shape)/self.gridRF.nCells)   
            for d in range(1,int(np.prod(self.theta_c.shape)/self.gridRF.nCells)+1):
                self.pParams['p_gamma'][d-1]=[]
        for d in range(1,int(np.prod(self.theta_c.shape)/self.gridRF.nCells)+1):
            self.pParams['p_gamma'][d-1].append(self.gamma[d-1,0])
            sb5.plot(np.arange(1,self.EM_iter+1),self.pParams['p_gamma'][d-1],color=colors[(d % 6),:])
        
        sb5.set_xlabel('iter')
        sb5.set_ylabel(r'$\gamma$')
        sb5.set_yscale('log')
        

        sb6 = plt.subplot(3, 2, 6)
        
        im = sb6.imshow(np.sqrt(self.sigma_cf['s0']).reshape((nSX, nSY),order='F'),origin='lower')
        sb6.set_title(r'$\sigma_{cf}$')
        plt.colorbar(im)
        sb6.set_xticks([])
        sb6.set_yticks([])
        sb6.grid(False)
        
        plt.savefig(path+'/state/params_epoch='+str(self.EM_iter)+'.eps')

    def plotElbo(self,path):

        plt.figure()
        self.Elbox.append(self.EM_iter)
        self.Elboy.append(self.elbo)
        plt.plot(self.Elbox,self.Elboy)
        plt.savefig(path+'/state/Elbo_epoch='+str(self.EM_iter)+'.eps')
    
    def plotCellScores(self,path):
        
        plt.figure()
        im=plt.imshow(np.reshape(self.rf2fem@(-self.cell_score),(np.prod(self.coarseGridX.shape), np.prod(self.coarseGridY.shape)),order='F').T,origin='lower')
        plt.colorbar(im)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title('Elbo cell score')
        plt.savefig(path+'/state/cellscore_epoch='+str(self.EM_iter)+'.eps')

    def write2file(self, path,params=[]):
       
        if params=='elbo':
            filename = path+'/savedata/elbo.mat'
            elbo = self.elbo
            scio.savemat(filename, {'elbo':elbo})

        if params=='cell_score':
            filename = path+'/savedata/cell_score.mat'
            cell_score = self.cell_score.T
            scio.savemat(filename, {'cell_score':cell_score})
      
        if params=='cell_score_full':
            filename = path+'/savedata/cell_score_full.mat'
            cell_score_full = self.cell_score_full.T
            scio.savemat(filename, {'cell_score_full':cell_score_full})
    
        if params=='sigma_cf_score':
            filename = path+'/savedata/sigma_cf_score.mat'
            sigma_cf_score = self.sigma_cf_score.T
            scio.savemat(filename, {'sigma_cf_score':sigma_cf_score})
        
        if params=='inv_sigma_cf_score':
            filename = path+'/savedata/inv_sigma_cf_score.mat'
            inv_sigma_cf_score = self.inv_sigma_cf_score.T
            scio.savemat(filename, {'inv_sigma_cf_score':inv_sigma_cf_score})
        
        
        
        if params=='thetaPriorHyperparam':
            filename = path+'/savedata/thetaPriorHyperparam.mat'
            thetaPriorHyperparam = self.gamma.T
            scio.savemat(filename, {'thetaPriorHyperparam':thetaPriorHyperparam})
        
        
        if params=='theta_c':
            filename = path+'/savedata/theta_c.mat'
            tc = self.theta_c.T
            scio.savemat(filename, {'tc':tc})
        
        
        if params=='sigma_theta_c':
            filename = path+'/savedata/Sigma_theta_c.mat'
            Sigma_theta_c = self.Sigma_theta_c
            scio.savemat(filename, {'Sigma_theta_c':Sigma_theta_c})
        
        
        if params=='sigma_c':
            filename = path+'/savedata/sigma_c.mat'
            sc = np.diag(self.Sigma_c.todense()).T
            scio.savemat(filename, {'sc':sc})
        
        
        if params=='sigma_cf':
            filename = path+'/savedata/sigma_cf.mat'
            scf = self.sigma_cf['s0'].T
            scio.savemat(filename, {'scf':scf})
        
        
        if params=='vardist':
            varmu = self.variational_mu
            varsigma = self.variational_sigma
            scio.savemat(path+'/savedata/vardistparams.mat', {'varmu':varmu,'varsigma':varsigma})
       


        