import numpy as np
import os
import scipy
import scipy.io as scio
import scipy.interpolate
import matplotlib.pyplot as plt
import time
from threading import Thread
from Meshpy import RectangularMesh
from featurefunctionpy import *
import warnings
from Mythreadpy import MyThread

class StokesData:
    def __init__(self, samples,path):
        
        self.meshSize = 256
        self.numberParams = [7.8, 0.2]   
        self.numberDist = 'logn'
        self.margins = [0.003, 0.003, 0.003, 0.003]    
        self.r_params = [-5.23, 0.3]    
        self.coordDist = 'GP'
        self.coordDist_mu = '0.5_0.5'   
        self.coordDist_cov = 'squaredExponential'
        self.radiiDist = 'lognGP'
        self.densityLengthScale = '0.08'
        self.sigmoidScale = '1.2'
        self.sigmaGP_r = 0.4
        self.l_r = 0.05
        self.origin_rejection = False
        self.samples = samples
        self.nSamples = len(samples)
        self.path=path
        self.pathname = []
        self.X = []
        self.X_interp = []
        self.input_bitmap = []
        self.P = []
        self.U = []
        self.bc = []
        self.cells = []
        self.cellOfVertex = []
        self.N_vertices_tot = []
        self.microstructData = []
        self.p_bc = '0.0'
        self.u_bc = ['u_x=1.0-0.0x[1]', 'u_y=1.0-0.0x[0]']
        self.a_x_m = []
        self.a_x_s = 1.0
        self.a_y_m = 0.0
        self.a_y_s = 1.0
        self.a_xy_m = 0.0
        self.a_xy_s = 1.0
        self.designMatrix = []
        self.designMatrixSqSum = []
    
    def setPathName(self):
        if not self.pathname:
            self.pathname = self.path+"/data/"
            
    def readData(self, quantities):
        
        print('Reading data from disk...')
        self.setPathName()

        cellIndex = 1
        for n in self.samples:
            if len(self.a_x_m)==0:
                foldername = self.pathname+'p_bc='+str(self.p_bc)+'/'+str(self.u_bc[0])+'_'+str(self.u_bc[1])
            else:
                foldername = self.pathname+'p_bc='+str(self.p_bc)+'/a_x_m='+str(self.a_x_m)+'_a_x_s='+str(self.a_x_s)+'a_y_m='+str(self.a_y_m)+'_a_y_s='+str(self.a_y_s)+'a_xy_m='+str(self.a_xy_m)+'_a_xy_s='+str(self.a_xy_s)
            filename = foldername+'/solution'+str(n)+'.mat'
            if os.path.isfile(filename):
                file = scio.loadmat(filename)
                if 'x' in quantities:
                    self.X.append(file['x'])
                if 'p' in quantities:
                    self.P.append(file['p'].T)
                    if self.a_x_m:
                        self.bc.append(file['bc'])

                if 'u' in quantities:
                    self.U.append(file['u'])

                if 'c' in quantities:
                    cellfile = scio.loadmat(f"{self.pathname}mesh{n}.mat")
                    self.cells.append(cellfile['cells'])

                if 'm' in quantities:
                    datafile = f"{self.pathname}microstructureInformation{n}.mat"
                    self.microstructData.append(scio.loadmat(datafile))

                cellIndex = cellIndex + 1
            else:
                self.samples.remove(n)
                self.nSamples = self.nSamples - 1
                print(f"{filename} not found. Skipping sample.")     
        if  len(self.designMatrix)==0:
            self.designMatrix = (None,) * (cellIndex - 1)
        print('... data loaded to workspace.')

    def countVertices(self):
        self.N_vertices_tot = 0
        if len(self.P)==0:
            self.readData('p')
        for cellIndex in range(len(self.P)):
            self.N_vertices_tot += len(self.P[cellIndex])
    
    def interpolate(self, modelParams):
        
        fineGridX = np.cumsum(np.append(0,modelParams.fineGridX))
        fineGridY = np.cumsum(np.append(0,modelParams.fineGridY))
        
        if len(self.X)==0:
            self.readData('x')

        xq, yq = np.meshgrid(fineGridX, fineGridY)

        for n in range(len(self.P)):
            if len(self.P[n]) !=0: 
                cartcoord = list(zip(self.X[n][:, 0], self.X[n][:, 1]))
                F = scipy.interpolate.LinearNDInterpolator(cartcoord, np.squeeze(self.P[n]))                     
                p_interp = F(xq, yq)
                p_interp=np.ravel(p_interp, order='F')
                self.P[n]=p_interp
                
        mh1=np.vstack(xq.flatten('F'))
        mh2=np.vstack(yq.flatten('F'))
        self.X_interp=[]
        self.X_interp.append(np.hstack((mh1,mh2)))

    def shiftData(self, interp, quantity='p', point=[0, 0], value=0):
        
        if 'p' in quantity:
            for n in range(len(self.P)):
                if interp:
                    dist = np.sum((self.X_interp[0] - point)**2, axis=1)
                    
                else:
                    dist = np.sum((self.X[n] - point)**2, axis=1)
                p_temp = self.P[n]
                min_dist_i = np.argmin(dist)
                p_point = p_temp[min_dist_i]
                p_temp = p_temp - p_point + value
                self.P[n] = p_temp
        else:
            raise ValueError('shifting only implemented for P')
    
    def vtx2Cell(self, modelParams):

        cumsumX = np.cumsum(modelParams.fineGridX)
        cumsumX[-1] = cumsumX[-1] + 1e-12  
        cumsumY = np.cumsum(modelParams.fineGridY)
        cumsumY[-1] = cumsumY[-1] + 1e-12  
        
        Nx = len(modelParams.fineGridX)
        
        if(len(self.X) == 0 and len(self.X_interp) == 0):
            self = self.readData('x')
        if any(modelParams.interpolationMode):
            if len(self.X_interp) == 0:
                self = self.interpolate(modelParams)
            X = self.X_interp
        else:
            X = self.X
        self.cellOfVertex=[]
        for n in range(len(X)):
            self.cellOfVertex.append(np.zeros((X[n].shape[0], 1)))
            for vtx in range(X[n].shape[0]):
                nx = 1
                while(X[n][vtx, 0] > cumsumX[nx - 1]):
                    nx = nx + 1
                ny = 1
                while(X[n][vtx, 1] > cumsumY[ny - 1]):
                    ny = ny + 1
                self.cellOfVertex[n][vtx] = nx + (ny - 1)*Nx

    def input2bitmap(self, resolution=256):
        
        if len(self.microstructData)==0:
            self.readData('m')
        
        xx, yy = np.meshgrid(np.linspace(0, 1, resolution),np.linspace(0, 1, resolution))
        self.input_bitmap=[]
        for n in range(self.nSamples):
            r2 = self.microstructData[n]['diskRadii']**2
            self.input_bitmap.append(np.zeros((resolution, resolution), dtype=bool))
        
            for nCircle in range(len(self.microstructData[n]['diskRadii'][0])):               
                self.input_bitmap[n]=self.input_bitmap[n] | \
                ((xx - self.microstructData[n]['diskCenters'][nCircle, 0])**2 +\
                (yy - self.microstructData[n]['diskCenters'][nCircle, 1])**2 <= r2[0,nCircle])


    def bxFeature(self,n, lambdaMat,mData,gridRF,path): 

        dMat=[]
        delta_log=1
        grid1x1 = RectangularMesh(1)
        M1 = grid1x1.map2fine(gridRF)
        
        dMat=np.ones((gridRF.nCells, 1))
        if n==1:
            with open(path+'/data/features.txt', 'w') as file:
                file.write('const')
        
        phi=volumeFractionCircExclusions(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1)
        dMat=np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('poreFraction1x1')
        
    
        dMat=np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logporeFraction1x1')

        dMat=np.concatenate((dMat, M1*np.sqrt(phi)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('sqrtporeFraction1x1')
        
        dMat=np.concatenate((dMat, M1*(phi**1.5)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('PoreFraction^1.5_1x1')

        dMat=np.concatenate((dMat, M1*(phi**2)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('PoreFraction^2_1x1')

        dMat=np.concatenate((dMat, M1*(phi**2.5)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('PoreFraction^2.5_1x1')
        
        dMat=np.concatenate((dMat, M1*np.exp(phi)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('expPoreFraction1x1')
        
        dMat=np.concatenate((dMat, M1*np.log(np.abs(2*phi-1)+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log_SCA1x1')

        dMat=np.concatenate((dMat, M1*phi/(2-phi)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('maxwellApproximation1x1')
        
        dMat=np.concatenate((dMat, M1*np.log(phi/(2-phi)+ delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log_maxwellApproximation1x1')
        
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .005)

        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens005_1x1')

        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .0025)

        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens0025_1x1')

        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .00125)

        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens00125_1x1')

        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .000625)

        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens000625_1x1')
        
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .0003)
        
        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens0003_1x1')
            
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, 0)
        
        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens0_1x1')

        
        phi = interfacePerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('interfaceArea_1x1')
        
        
        dMat = np.concatenate((dMat, M1*np.log(phi+ delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('loginterfaceArea_1x1')
        
        
        dMat = np.concatenate((dMat, M1*(np.abs(np.log(phi+ delta_log))**1.5)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('abs(loginterfaceArea)^1.5_1x1')
        
        
        dMat = np.concatenate((dMat, M1*(np.log(phi+ delta_log)**2)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^2interfaceArea_1x1')    

        
        dMat = np.concatenate((dMat, M1*(np.log(phi+ delta_log)**3)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^3interfaceArea_1x1')          
        
        
        dMat = np.concatenate((dMat, M1*(np.log(phi+ delta_log)**4)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^4interfaceArea_1x1')   

        
        dMat = np.concatenate((dMat, M1*(np.abs(np.log(phi+ delta_log))**.5)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^1/2interfaceArea_1x1') 
        
        
        dMat = np.concatenate((dMat, M1*(np.abs(np.log(phi+ delta_log))**(1/3))), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^1/3interfaceArea_1x1') 
        
        
        dMat = np.concatenate((dMat, M1*(np.abs(np.log(phi+ delta_log))**(1/4))), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^1/4interfaceArea_1x1') 

        
        dMat = np.concatenate((dMat, M1*np.sqrt(phi)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('sqrtinterfaceArea_1x1') 
        
        
        dMat = np.concatenate((dMat, M1*phi**(1/3)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('interfaceArea^(1/3)_1x1') 
        
        
        dMat = np.concatenate((dMat, M1*phi**(1/4)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('interfaceArea^(1/4)_1x1') 
        
       
        dMat = np.concatenate((dMat, M1*phi**(1/5)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('interfaceArea^(1/5)_1x1') 

        
        dMat = np.concatenate((dMat, M1*phi**(2)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareinterfaceArea_1x1')

        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, 'mean','edge2edge')
        dMat = np.concatenate((dMat, M1*phi), axis=1) 
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('meanDist_1x1')
        
        
        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logMeanDist_1x1')
        
        
        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)**2), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squarelogMeanDist_1x1')

         
        dMat = np.concatenate((dMat, M1*np.log(phi+delta_log)**3), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^3MeanDist_1x1')

        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, 'mean', 2)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('meanDistCenter_1x1')
        
        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, 'min', 2)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('minDistCenter_1x1')

        
        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logminDistCenter_1x1')
        
        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)**2), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logminDistCentersq_1x1')
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .25)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath25_1x1')

        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath25_1x1')
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .1)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath1_1x1')            

        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath1_1x1')
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .05)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath05_1x1')            

        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath05_1x1')   

        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .02)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath02_1x1')            

        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath02_1x1')  

        _, h_v, poreSizeDens,_ =voidNearestSurfaceExclusion(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, 0)       
        dMat = np.concatenate((dMat, M1*h_v), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('h_v0_1x1')   
        
        dMat = np.concatenate((dMat, M1*poreSizeDens), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('poreSizeProbDens0_1x1') 
        
        dMat = np.concatenate((dMat, M1*np.log(h_v+delta_log),M1*np.log(poreSizeDens+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log_h_v0_1x1')     
                file.write('\n') 
                file.write('log_poreSizeProbDens0_1x1') 
        
        phi = meanChordLength(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('meanChordLength_1x1') 


        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logMeanChordLength_1x1')  
        
        dMat = np.concatenate((dMat, M1*np.exp(phi)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('expmeanChordLength_1x1')   

        dMat = np.concatenate((dMat, M1*np.sqrt(phi)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('sqrtmeanChordLength_1x1')

       
        phi = momentPerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .2)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('0.2_moment')

        
        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log0.2_moment') 

        phi = momentPerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, .5)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('0.5_moment')


        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log0.5_moment') 
        
        phi = momentPerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], grid1x1, 1.0)
        dMat = np.concatenate((dMat, M1*phi), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('1.0_moment')

        
        dMat = np.concatenate((dMat, np.log(M1*phi+delta_log)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log1.0_moment')

        
        phi = volumeFractionCircExclusions(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('poreFraction')

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logporeFraction')         
        
         
        dMat = np.concatenate((dMat, np.sqrt(phi).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('sqrtporeFraction')

        
        dMat = np.concatenate((dMat, (phi**1.5).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('PoreFraction^1.5')
        
        dMat = np.concatenate((dMat, (phi**2).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('PoreFraction^2')
        
        dMat = np.concatenate((dMat, (phi**2.5).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('PoreFraction^2.5')


        dMat = np.concatenate((dMat, np.exp(phi).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('expporeFraction')
        
        
        dMat = np.concatenate((dMat, np.log(np.abs(2*phi-1)+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log_SCA')

        
        dMat = np.concatenate((dMat, (phi/(2-phi)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('maxwellApproximation')
        
        
        dMat = np.concatenate((dMat, np.log((phi/(2-phi))+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n')
                file.write('log_maxwellApproximation')
        
        
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .005)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens005')
        
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .0025)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens0025')

        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .00125)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens00125')

        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .000625)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens000625')

        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .0003)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens0003')
        
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .00015)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens00015')
        
        phi = chordLengthDensity(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .0)
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logChordLengthDens0')
        
        
        phi = interfacePerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('interfaceArea')

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logInterfaceArea')

        
        dMat = np.concatenate((dMat, (np.abs(np.log(phi+delta_log))**1.5).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('abs(logInterfaceArea)^1.5')
        
        
        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**2).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^2InterfaceArea')
        
        
        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**3).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^3InterfaceArea')
        
        
        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**4).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^4InterfaceArea')
        
        
        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**5).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^5InterfaceArea')
        
        
        dMat = np.concatenate((dMat, (np.abs(np.log(phi+delta_log))**0.5).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^1/2InterfaceArea')
        
        
        dMat = np.concatenate((dMat, (np.abs(np.log(phi+delta_log))**(1/3)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^1/3InterfaceArea')
        
        
        dMat = np.concatenate((dMat, (np.abs(np.log(phi+delta_log))**(1/4)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^1/4InterfaceArea')

        
        dMat = np.concatenate((dMat, np.sqrt(phi).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n')
                file.write('sqrtInterfaceArea')
        
        
        dMat = np.concatenate((dMat, (phi**(1/3)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('InterfaceArea^(1/3)')
        
        
        dMat = np.concatenate((dMat, (phi**(1/4)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('InterfaceArea^(1/4)')

        
        dMat = np.concatenate((dMat, (phi**(1/5)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('InterfaceArea^(1/5)')
        
        
        dMat = np.concatenate((dMat, (phi**(2)).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareInterfaceArea')
        
        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'mean','edge2edge')
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('meanDist')
        
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logmeanDist') 

           
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'max','edge2edge')
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('maxDist') 

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logMaxDist')

           
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'std','edge2edge')
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('stdDist') 

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n')
                file.write('logstdDist')

        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'squareWellPot','edge2edge', .01)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)  
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareWellPot01') 
        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'squareWellPot','edge2edge', .02)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)  
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareWellPot02')

        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'squareWellPot','edge2edge', .03)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)  
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareWellPot03')

        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'squareWellPot','edge2edge', .04)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)  
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareWellPot04')

        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'squareWellPot','edge2edge', .05)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)  
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareWellPot05')
        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'mean','edge2edge')
        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**2).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('squareInterfaceArea')
        

        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**3).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log^3InterfaceArea')

        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'mean', 2)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('meanDistCenter')  

        
        phi = diskDistance(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 'min', 2)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('minDistCenter') 

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logMinDistCenter')
        
        
        dMat = np.concatenate((dMat, (np.log(phi+delta_log)**2).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logMinDistCenterSq')
        
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .25)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath25') 
        
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath25')
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .1)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath1')
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath1')
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .05)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath05')
        
        #log
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath05')
        
        phi = matrixLinealPath(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .02)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath1')
        
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logLinPath02')
        
        
        _, h_v, poreSizeDens, _ = voidNearestSurfaceExclusion(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 0)
        dMat = np.concatenate((dMat, h_v.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('h_v0')

        dMat = np.concatenate((dMat, poreSizeDens.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('poreSizeProbDens0')    

        
        dMat = np.concatenate((dMat, np.log(h_v+delta_log).reshape(-1,1),np.log(poreSizeDens+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log_h_v0')
                file.write('\n') 
                file.write('log_poreSizeProbDens0')        

        
        phi = meanChordLength(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPath1')
        
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('logMeanChordLength')

        
        dMat = np.concatenate((dMat, np.exp(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('expMeanChordLength')

        
        dMat = np.concatenate((dMat, np.sqrt(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('sqrtMeanChordLength')            

        
        phi = momentPerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .2)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('0.2_moment')
        
        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log0.2_moment')

        phi = momentPerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, .5)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('0.5_moment')

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log0.5_moment')

        phi = momentPerVolume(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, 1.0)
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('1.0_moment')

        
        dMat = np.concatenate((dMat, np.log(phi+delta_log).reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('log1.0_moment')

        phi= linPathLengthScale(mData[n-1]['diskCenters'],mData[n-1]['diskRadii'], gridRF, [0, .02, .05, .1, .25])
        dMat = np.concatenate((dMat, phi.reshape(-1,1)), axis=1)
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('linPathLengthScale')

        
        phi_tmp = np.zeros((gridRF.nCells,13))
        for k in range(1,gridRF.nCells+1):       
            phi_tmp[k-1, 0]= distanceTransform(lambdaMat[n-1, k-1], 'euclidean', False, 'mean')
            phi_tmp[k-1, 1]= distanceTransform(lambdaMat[n-1, k-1], 'euclidean', False, 'var')      
            phi_tmp[k-1, 2]= distanceTransform(lambdaMat[n-1, k-1], 'euclidean', False, 'max')
            phi_tmp[k-1, 3]= distanceTransform(lambdaMat[n-1, k-1], 'chessboard', False, 'mean')  
            phi_tmp[k-1, 4] = distanceTransform(lambdaMat[n-1, k-1], 'chessboard', False, 'var')           
            phi_tmp[k-1, 5] = distanceTransform(lambdaMat[n-1, k-1], 'chessboard', False, 'max')                    
            phi_tmp[k-1, 6] = distanceTransform(lambdaMat[n-1, k-1], 'cityblock', False, 'mean')
            phi_tmp[k-1, 7] = distanceTransform(lambdaMat[n-1, k-1], 'cityblock', False, 'var')   
            phi_tmp[k-1, 8] = distanceTransform(lambdaMat[n-1, k-1], 'cityblock', False, 'max')
            phi_tmp[k-1, 9] = gaussLinFilt(lambdaMat[n-1, k-1], np.nan, 2)
            phi_tmp[k-1, 10] = gaussLinFilt(lambdaMat[n-1, k-1], np.nan, 5)
            phi_tmp[k-1, 11] = gaussLinFilt(lambdaMat[n-1, k-1], np.nan, 10)            
            phi_tmp[k-1, 12] = isingEnergy(lambdaMat[n-1, k-1])           
            
        dMat = np.concatenate((dMat, phi_tmp), axis=1)
        
        
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformEuclideanMean') 
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformEuclideanVar') 
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformEuclideanMax')             
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformChessboardMean') 
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformChessboardVar') 
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformChessboardMax') 
        
        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformCityblockMean')

        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformCityblockVar')

        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('distTransformCityblockMax')  

        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('gaussLinFilt2') 

        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('gaussLinFilt5')  

        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('gaussLinFilt10')

        if n == 1:
            with open(path+'/data/features.txt', 'a') as file:
                file.write('\n') 
                file.write('isingEnergy')  
        
        return dMat


    def evaluateFeatures(self, gridRF,path):
        
        if len(self.microstructData)==0:
            self.readData('m')   

        resolution=512
        if len(self.input_bitmap)==0:
            self.input2bitmap(resolution)

        mData = self.microstructData
        indicator, nrow, ncol = gridRF.indexIndicator(resolution)

        lambdaMat=np.array([[None]*(gridRF.nCells) for i in range(len(self.samples))])
        for n in range(1,len(self.samples)+1):
            for k in range(1,gridRF.nCells+1):
                lambdaMat[n-1,k-1]=np.reshape(self.input_bitmap[n-1][indicator[k-1]],(nrow[k-1],ncol[k-1]))
        print('Evaluating feature functions...')

        dMat=self.designMatrix
        threads = []
        
        for n in range(1,len(self.samples)+1):
            thread = MyThread(self.bxFeature,(n,lambdaMat,mData,gridRF,path))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        i=0
        for thread in threads:
            a=thread.get_result()
            dMat[i]=a
            i=i+1
        self.designMatrix = dMat
        print('...feature functions evaluated.')

    def computeFeatureFunctionMinMax(self):
        
        featFuncMin = self.designMatrix[0].copy()
        featFuncMax = self.designMatrix[0].copy()
        for n in range(1,len(self.designMatrix)+1):
            featFuncMin[featFuncMin > self.designMatrix[n-1]] =self.designMatrix[n-1][featFuncMin > self.designMatrix[n-1]]
            featFuncMax[featFuncMax < self.designMatrix[n-1]] =self.designMatrix[n-1][featFuncMax < self.designMatrix[n-1]]
        return featFuncMin, featFuncMax


    def rescaleDesignMatrix(self,path,featFuncMin=[], featFuncMax=[]):
        
        print('Rescale design matrix...')
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        if len(featFuncMin)==0 or len(featFuncMax)==0:
            featFuncMin, featFuncMax = self.computeFeatureFunctionMinMax()
        featFuncDiff = featFuncMax - featFuncMin
        
        sameOutput = featFuncDiff == 0
        for n in range(1,len(self.designMatrix)+1):
            self.designMatrix[n-1] =(self.designMatrix[n-1] - featFuncMin)/(featFuncDiff)
            self.designMatrix[n-1][sameOutput] = 1
        
        
        self.saveNormalization('rescaling', featFuncMin, featFuncMax,path)
        print('done')
                       
    def saveNormalization(self, type, a, b,path):
            print('Saving design matrix normalization...')
            if type== 'standardization':
                scio.savemat(path,{'featFuncMean':a,'featFuncSqMean':b})
            elif type=='rescaling':
                scio.savemat(path,{'featFuncMin':a,'featFuncMax':b})
            else:
                raise ValueError('Which type of data normalization?')
            
    def shapeToLocalDesignMat(self):
        debug = False  
        print("Using separate feature coefficients theta_c for each macro-cell in a microstructure...")

        nElc, nFeatureFunctions = self.designMatrix[0].shape  
        Phi = [np.zeros((nElc, nElc * nFeatureFunctions)) for _ in range(len(self.designMatrix))]  
        
        for n in range(1,len(self.designMatrix)+1):
            for k in range(1,nElc+1):
                Phi[n-1][k-1, (k-1)*nFeatureFunctions:k* nFeatureFunctions] = self.designMatrix[n-1][k-1, :]
            Phi[n-1] = scipy.sparse.csr_matrix(Phi[n-1])

        if debug:
            first_design_matrix_before_local = self.designMatrix[0]
            first_design_matrix_after_local = np.full((nElc, nElc * nFeatureFunctions), 0)
            first_design_matrix_after_local[:, :] = Phi[0].todense()
            print("firstDesignMatrixBeforeLocal:\n", first_design_matrix_before_local)
            print("firstDesignMatrixAfterLocal:\n", first_design_matrix_after_local)
            input("Press Enter to continue...")

        self.designMatrix = Phi
        print("done")