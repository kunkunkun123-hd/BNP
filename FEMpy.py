import numpy as np
import time
import scipy.sparse as sp
import scipy.io as scio
import scipy.integrate as integrate

def get_loc_force_gradient2(domain, d_k,el):
    
    f2 = np.zeros(4)

    Tb = np.zeros(4)
    Tbflag = False
    
    globNode = domain.globalNodeNumber[el-1, :]
    for i in range(1,4+1):
        if ~np.isnan(domain.essentialTemperatures[globNode[i-1]]):
            Tb[i-1] = domain.essentialTemperatures[globNode[i-1]]
            Tbflag = True
    if Tbflag:
        f2 = -d_k@Tb
    return f2

def get_glob_force_gradient(domain, d_k, el):
    
    F = np.zeros(int(domain.nEq))
    f = get_loc_force_gradient2(domain, d_k,el)

    ln = np.arange(1,4+1)
    eqnTemp = domain.lm[el-1, :]
    eqn = eqnTemp[eqnTemp > 0]
    ln = ln[eqnTemp > 0]
    F[eqn-1] = F[eqn-1] + f[ln-1]
    return F

def get_glob_stiff3(d_Ke, conductivity, nEq):

    K = sp.csr_matrix(np.reshape(d_Ke@conductivity,(int(nEq), int(nEq)),order='F'))

    return K

def get_glob_force(mesh, conductivity):

    F = mesh.F_natural
    
    anyEssentialNodeInElement = np.any(mesh.essentialNodeInElement.T,axis=0)
    for e in range(1,mesh.nEl+1):

        if anyEssentialNodeInElement[e-1]:
            Tb = np.zeros((4, 1))
            for i in range(1,5):
                if mesh.essentialNodeInElement[e-1, i-1]:
                    Tb[i-1] = mesh.essentialTemperatures[mesh.globalNodeNumber[e-1, i-1]]

            for ln in range(1,5):
                eqn = mesh.lm[e-1, ln-1]
                if(eqn != 0):

                    fT = conductivity[e-1]*mesh.d_loc_stiff[:, :, e-1]@Tb
                    F[eqn-1] = F[eqn-1] - fT[ln-1]
    return F

def heat2d(mesh, conductivity):
    
    Out={}
    Out['globalStiffness'] = get_glob_stiff3(mesh.d_glob_stiff_assemble, conductivity, mesh.nEq)
    
    Out['naturalTemperatures'] = np.linalg.solve(Out['globalStiffness'].todense(),get_glob_force(mesh,conductivity))
    Out['u'] = np.zeros((mesh.nNodes, 1))
    Out['u'][mesh.id,0]=Out['naturalTemperatures']
    Out['u'][mesh.essentialNodes-1, 0] = mesh.essentialTemperatures[mesh.essentialNodes-1]
    return Out

def FEMgrad(u, mesh):
    
    d_r = np.reshape(mesh.d_glob_stiff@u, (int(mesh.nEq), int(mesh.nEl)),order='F').T- mesh.d_glob_force

    return d_r

def get_adjoints(K, WTSinv, domain, Tf_i_minus_mu_minus_WTc):
    
    d_log_p_cf = WTSinv@Tf_i_minus_mu_minus_WTc
    
    d_log_p_cf=d_log_p_cf[np.isnan(domain.essentialTemperatures)]

    adjoints = np.linalg.solve(K.todense(), d_log_p_cf)
    return adjoints

class MeshFEM:
    def __init__(self,gridX, gridY):
        self.gridX=gridX  
        self.gridY=gridY
        self.nElX=len(gridX)  
        self.nElY=len(gridY)
        self.nEl=self.nElX*self.nElY  
        self.nNodes=[]  
        self.boundaryNodes=[]  
        self.essentialNodes=[]  
        self.essentialTemperatures=[]  
        self.naturalNodes=[]  
        self.boundaryElements=[]  
        self.naturalBoundaries=[]  
        self.boundaryType=[]  
        self.lx=1  
        self.ly=1
        self.lElX=[] 
        self.lElY=[]  
        self.cum_lElX=[]  
        self.cum_lElY=[]
        self.AEl=[]  
        self.nEq=[]  
        self.lc=[]  
        self.nodalCoordinates=[]  
                            
        self.globalNodeNumber=[]  
                            
        self.essentialNodeInElement=[]
        self.Bvec=[]  
        self.d_N=[] 
        self.NArray=[]
        self.convectionMatrix=[]  
        self.essentialBoundary=[] 
        self.lm=[]  
            
        self.id=[]  
        self.Equations=[]  
        self.LocalNode=[]
        self.kIndex=[]

        self.fs=[]  
        self.fh=[]  
        self.f_tot=[]  
        self.F_natural=[]  

        self.compute_grad = False
        self.d_loc_stiff=[]  
        self.d_glob_stiff=[]  
        self.d_glob_stiff_assemble=[]    
        
        self.d_glob_force=[]

        
        diffX=abs(sum(gridX) - self.lx)
        diffY=abs(sum(gridY) - self.ly)
        assert diffX < np.finfo(float).eps, 'element lengths do not sum up to lx'
        assert diffY < np.finfo(float).eps, 'element lengths do not sum up to ly'

        self.lElX = np.zeros(self.nEl)
        self.lElY = np.zeros(self.nEl)
        self.AEl = np.zeros(self.nEl)
        for e in range(1,self.nEl+1):
            self.lElX[e-1]=gridX[np.mod(e-1,self.nElX)]
            self.lElY[e-1]=gridY[int(np.floor((e-1)/self.nElX))]
            self.AEl[e-1] = self.lElX[e-1]*self.lElY[e-1]
        
        self.cum_lElX = np.cumsum(np.concatenate(([0], gridX)))
        self.cum_lElY = np.cumsum(np.concatenate(([0], gridY)))
        self.nNodes = (self.nElX + 1)*(self.nElY + 1)
        self.boundaryNodes=np.array(list(range(1, self.nElX + 2)) +
                                   list(range(2 * (self.nElX + 1), (self.nElX + 1) * (self.nElY + 1) + 1,
                                            self.nElX + 1)) +
                                   list(range((self.nElX + 1) * (self.nElY + 1) - 1, (self.nElX +1) * self.nElY,
                                            -1)) +
                                   list((self.nElX + 1) * np.arange(self.nElY - 1, 0, -1) + 1),
                                   dtype=np.int32)
        self.boundaryElements = np.array(list(range(1, self.nElX + 1)) +
                                      list(range(2 * self.nElX, self.nElX * self.nElY + 1, self.nElX)) +
                                      list(range(self.nElX * self.nElY - 1, self.nElX * (self.nElY - 1), -1)) +
                                      list(self.nElX * np.arange(self.nElY - 2, 0, -1) + 1),
                                      dtype=np.int32)
        self.setLocCoord()
        self.setGlobalNodeNumber()

        self.setHeatSource(np.zeros(self.nEl))  
    

    def setLocCoord(self):
    
        self.lc = np.zeros((self.nEl, 4, 2))
        for e in range(1,self.nEl+1):
            row = int(np.floor((e-1) / self.nElX))
            col = np.mod((e-1), self.nElX)

            self.lc[e-1, 0, 0] = self.cum_lElX[col]
            self.lc[e-1, 1, 0] = self.cum_lElX[col+1]
            self.lc[e-1, 2, 0] = self.lc[e-1, 1, 0]
            self.lc[e-1, 3, 0] = self.lc[e-1, 0, 0]

            self.lc[e-1, 0, 1] = self.cum_lElY[row]
            self.lc[e-1, 1, 1] = self.lc[e-1, 0, 1]
            self.lc[e-1, 2, 1] = self.cum_lElY[row+1]
            self.lc[e-1, 3, 1] = self.lc[e-1, 2, 1]

    def setGlobalNodeNumber(self):
        self.globalNodeNumber = np.zeros((self.nEl, 4), dtype=np.int32)
        for e in range(1,self.nEl+1):
                self.globalNodeNumber[e-1, 0] = e + np.floor((e-1)/self.nElX)-1
                self.globalNodeNumber[e-1, 1] = e + np.floor((e-1)/self.nElX)+1-1
                self.globalNodeNumber[e-1, 2] = self.globalNodeNumber[e-1, 0] + self.nElX + 2
                self.globalNodeNumber[e-1, 3] = self.globalNodeNumber[e-1, 0] + self.nElX + 1

    def setHeatSource(self, heatSourceField):
       
        xi1 = -1/np.sqrt(3)
        xi2 = 1/np.sqrt(3)
        eta1 = -1/np.sqrt(3)
        eta2 = 1/np.sqrt(3)
            
        self.fs = np.zeros((4, self.nEl))

        for e in range(1,self.nEl+1):
            x1 = self.lc[e-1, 0, 0]
            x2 = self.lc[e-1, 1, 0]
            y1 = self.lc[e-1, 0, 1]
            y2 = self.lc[e-1, 3, 1]
                
            xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1)
            xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1)
            yI = 0.5*(y1 + y2) + 0.5*eta1*(y2 - y1)
            yII = 0.5*(y1 + y2) + 0.5*eta2*(y2 - y1)

            self.fs[0,e-1]=heatSourceField[e-1]*(1/self.AEl[e-1])*((xI - x2)*(yI - y2) + (xII - x2)*(yII - y2) + (xI - x2)*(yII - y2) + (xII - x2)*(yI - y2))
            self.fs[1, e-1] = -heatSourceField[e-1]*(1/self.AEl[e-1])*((xI - x1)*(yI - y2) + (xII - x1)*(yII - y2) + (xI - x1)*(yII - y2) + (xII - x1)*(yI - y2))
            self.fs[2, e-1] = heatSourceField[e-1]*(1/self.AEl[e-1])*((xI - x1)*(yI - y1) + (xII - x1)*(yII - y1) + (xI - x1)*(yII - y1) + (xII - x1)*(yI - y1))
            self.fs[3, e-1] = -heatSourceField[e-1]*(1/self.AEl[e-1])*((xI - x2)*(yI - y1) + (xII - x2)*(yII - y1) + (xI - x2)*(yII - y1) + (xII - x2)*(yI - y1))


    def elementShapeFunctions(self, x, y, xe, Ael, component=None):
        
        if component is None:
            N = np.zeros(4)
            N[0] = (x - xe[1]) * (y - xe[3])
            N[1] = -(x - xe[0]) * (y - xe[3])
            N[2] = (x - xe[0]) * (y - xe[2])
            N[3] = -(x - xe[1]) * (y - xe[2])
            N /= Ael
            return N
        else:
            if component == 1:
                N = (x - xe[1]) * (y - xe[3])
            elif component == 2:
                N = -(x - xe[0]) * (y - xe[3])
            elif component == 3:
                N = (x - xe[0]) * (y - xe[2])
            elif component == 4:
                N = -(x - xe[1]) * (y - xe[2])
            else:
                raise ValueError('Which local node?')
            N /= Ael
            return N


    def setFluxForce(self, qb):
                
        self.fh = np.zeros((4, self.nEl))
        xe=[None]*4
        for e in range(1,self.nEl+1):
            xe[0]=self.lc[e-1,0,0]
            xe[1]=self.lc[e-1,1,0]
            xe[2]=self.lc[e-1,0,1]
            xe[3]=self.lc[e-1,3,1]
            N=lambda x, y: self.elementShapeFunctions(x, y, xe, self.AEl[e-1])
            if e<=self.nElX and self.naturalBoundaries[e-1, 0]:
                
                q=lambda x:qb[0](x)
                Nlo = lambda x: N(x, 0)
                fun = lambda x: q(x)*Nlo(x)
                self.fh[:, e-1] += integrate.quad_vec(fun, xe[0], xe[1])[0]
            
            if np.mod(e, self.nElX) == 0 and self.naturalBoundaries[e-1, 1]:
                
                q=lambda y:qb[1](y)
                Nr=lambda y:N(1,y)
                fun=lambda y:q(y)*Nr(y)
                self.fh[:,e-1] += integrate.quad_vec(fun, xe[2], xe[3])[0]
            
            if e>(self.nElY-1)*self.nElX and self.naturalBoundaries[e-1,2]:
                
                q=lambda x:qb[2](x)
                Nu=lambda x:N(x,1)
                fun=lambda x:q(x)*Nu(x)
                self.fh[:,e-1] += integrate.quad_vec(fun, xe[0], xe[1])[0]

            if np.mod(e,self.nElX)==1 and self.naturalBoundaries[e-1, 3]:
                
                q=lambda y:qb[3](y)
                Nle=lambda y:N(0,y)
                fun=lambda y:q(y)*Nle(y)
                self.fh[:,e-1] += integrate.quad_vec(fun, xe[2], xe[3])[0]
        return self
    


    def getCoord(self):

        j = 1 
        self.nodalCoordinates = np.nan*np.zeros((3, self.nNodes))
        for i in range(1,self.nNodes+1):
        
            row = np.floor((i - 1)/(self.nElX + 1)) + 1
            col = np.mod((i - 1), (self.nElX + 1)) + 1
            
            x = self.cum_lElX[int(col-1)]
            y = self.cum_lElY[int(row-1)]

            self.nodalCoordinates[0, i-1] = x
            self.nodalCoordinates[1, i-1] = y
            
            if (self.essentialNodes == i).any():
               
                self.nodalCoordinates[2, i-1] = 0
            else:
                
                self.nodalCoordinates[2, i-1] = j
                j = j + 1 
        return self
    
    def setId(self):
        
        eqs, i = np.sort(self.nodalCoordinates[2, :]), np.argsort(self.nodalCoordinates[2, :])
        
        self.id = np.stack((eqs, i), axis=1)
        init = np.where(eqs == 1)[0][0]
        self.id=np.delete(self.id,range(init),axis=0)

        self.id = self.id[:, 1]
        self.id = self.id.astype(np.uint32)
        return self


    def getEquations(self):
        
        localNodeInit = np.arange(1,4+1)
        
        self.Equations = np.zeros((16*self.nEl, 2))
        self.LocalNode = np.zeros((16*self.nEl, 3))
        eq = 0 
        for e in range(1,self.nEl+1):
            equationslm = self.lm[e-1, localNodeInit-1]
            equations = equationslm[equationslm > 0]
            localNode = localNodeInit[equationslm > 0]
            prevnEq = eq
            eq = eq + len(equations)**2
        
            Equations1, Equations2 = np.meshgrid(equations,equations)
            self.Equations[prevnEq:eq, :] =np.column_stack([Equations1.ravel('F'), Equations2.ravel('F')])
   
            LocalNode1, LocalNode2 = np.meshgrid(localNode,localNode)
            self.LocalNode[prevnEq:eq, :] =np.column_stack([LocalNode1.ravel('F'), LocalNode2.ravel('F'), np.tile(e, (len(equations)**2, 1))])
        
        self.Equations=np.delete(self.Equations,np.s_[eq:],axis=0)
        self.LocalNode=np.delete(self.LocalNode,np.s_[eq:],axis=0)
        self.Equations = self.Equations.astype(np.uint32)
        self.LocalNode = self.LocalNode.astype(np.uint32)

        return self

    def setNodalCoordinates(self):
        self = self.getCoord()       
        self.lm = self.globalNodeNumber 
        for i in range(1,(self.globalNodeNumber).shape[0]+1):
            for j in range(1,(self.globalNodeNumber).shape[1]+1):
                self.lm[i-1, j-1] = self.nodalCoordinates[2, self.globalNodeNumber[i-1, j-1]]
                 
        self = self.setId()
        self = self.getEquations()
        
        self.Equations = np.double(self.Equations)
        self.kIndex = np.ravel_multi_index((self.LocalNode[:,0]-1, self.LocalNode[:,1]-1, self.LocalNode[:,2]-1),dims=(4, 4, self.nEl), order='F')
        
        return self

    def get_loc_stiff_grad(self):
        
        self.d_loc_stiff = np.zeros((4, 4, self.nEl))
        for e in range(1,self.nEl+1):
            self.d_loc_stiff[:, :, e-1] =(self.Bvec[:, :, e-1]).T@self.Bvec[:, :, e-1]
        
        return self

    def get_glob_stiff_grad(self):
        
        self.d_glob_stiff = []
        self.d_glob_stiff_assemble = np.zeros((int(self.nEq)*int(self.nEq), self.nEl))
        for e in range(1,self.nEl+1):
            grad_loc_k = np.zeros((4, 4, self.nEl))
            grad_loc_k[:, :, e-1] = self.d_loc_stiff[:, :, e-1]
            d_Ke =sp.csr_matrix((grad_loc_k[np.unravel_index(self.kIndex, grad_loc_k.shape,order='F')],((self.Equations[:, 0]).astype(np.int)-1,(self.Equations[:, 1]).astype(np.int)-1)))
            if e==1:                
                self.d_glob_stiff=d_Ke
            else:
                self.d_glob_stiff = sp.vstack([self.d_glob_stiff, d_Ke])
            self.d_glob_stiff_assemble[:, e-1] = d_Ke.todense().reshape((-1,),order='F')
                
        self.d_glob_stiff_assemble = sp.csr_matrix(self.d_glob_stiff_assemble)

        return self
    
    def get_glob_force_grad(self):
        
        self.d_glob_force=[]
        for e in range(1,self.nEl+1):
            self.d_glob_force.append(get_glob_force_gradient(self,self.d_loc_stiff[:, :, e-1], e))
        self.d_glob_force=np.array(self.d_glob_force)
        self.d_glob_force = sp.csr_matrix(self.d_glob_force)

        return self
    

    def setBvec(self):
        
        self.nEq = max(self.nodalCoordinates[2,:])
        
        xi1 = -1/np.sqrt(3)
        xi2 = 1/np.sqrt(3)
        
        self.Bvec = np.zeros((8, 4, self.nEl))
        self.essentialBoundary=np.zeros((4,self.nEl))
        for e in range(1,self.nEl+1):
            for i in range(1,5):
                self.essentialBoundary[i-1, e-1] =~np.isnan(self.essentialTemperatures[self.globalNodeNumber[e-1, i-1]])

            x1 = self.lc[e-1,0,0]
            x2 = self.lc[e-1,1,0]
            y1 = self.lc[e-1,0,1]
            y4 = self.lc[e-1,3,1]
  
            xI = 0.5*(x1 + x2) + 0.5*xi1*(x2 - x1)
            xII = 0.5*(x1 + x2) + 0.5*xi2*(x2 - x1)
            yI = 0.5*(y1 + y4) + 0.5*xi1*(y4 - y1)
            yII = 0.5*(y1 + y4) + 0.5*xi2*(y4 - y1)
           
            B1 = np.array([[yI-y4 ,y4-yI, yI-y1, y1-yI],[xI-x2, x1-xI, xI-x1, x2-xI]])
            B2 = np.array([[yII-y4, y4-yII, yII-y1, y1-yII],[xII-x2, x1-xII, xII-x1, x2-xII]])

            B3 = np.array([[yI-y4, y4-yI, yI-y1, y1-yI],[xII-x2, x1-xII, xII-x1, x2-xII]])
            B4 = np.array([[yII-y4, y4-yII, yII-y1, y1-yII],[xI-x2, x1-xI, xI-x1, x2-xI]])
        
            self.Bvec[:, :, e-1] = (1/(2*np.sqrt(self.AEl[e-1])))*np.vstack((B1,B2,B3,B4))

        self = self.get_loc_stiff_grad()
        if self.compute_grad:
            self = self.get_glob_stiff_grad()
            self = self.get_glob_force_grad()       
        return self
    def get_essential_node_in_element(self):
        
        self.essentialNodeInElement = False*np.ones((self.nEl, 4))
        for e in range(1,self.nEl+1):
            for i in range(1,4+1):
                self.essentialNodeInElement[e-1, i-1] =(self.globalNodeNumber[e-1, i-1] == (self.essentialNodes-1)).any()
        return self
    
    def get_glob_natural_force(self):
        
        self.F_natural = np.zeros(int(self.nEq))

        for e in range(1,self.nEl+1):
            for ln in range(1,4+1):
                eqn = self.lm[e-1, ln-1]        
                if eqn != 0:
                    self.F_natural[eqn-1] =self.F_natural[eqn-1] + self.f_tot[ln-1, e-1]
                     
        return self

    def setBoundaries(self, nat_nodes, Tb, qb):
        self.boundaryType = np.ones(2*self.nElX + 2*self.nElY, dtype=bool)
        self.boundaryType[np.array(nat_nodes)-1] = False
        self.essentialNodes = self.boundaryNodes[self.boundaryType]
        self.naturalNodes = np.int32(self.boundaryNodes[~self.boundaryType])

        self.essentialTemperatures = np.nan*np.ones((self.nNodes,))
        boundaryCoordinates0=[0.] + np.cumsum(self.lElX[0:self.nElX]).tolist() + (self.nElY-1)*[1.0] + np.flipud(np.cumsum(self.lElX[0:self.nElX])).tolist()+[0.]+[0.]*(self.nElY-1)
        boundaryCoordinates1=[0.]*(self.nElX+1)+np.cumsum(self.lElY[np.arange(self.nElX-1,(self.nElX*self.nElY),self.nElX)]).tolist()+[1.]*(self.nElX-1)+np.flipud(np.cumsum(self.lElY[np.arange(self.nElX-1,(self.nElX*self.nElY),self.nElX)])).tolist()     
        boundaryCoordinates = np.array([boundaryCoordinates0,boundaryCoordinates1])
        Tess = np.zeros((self.nNodes,)) 
        for i in range(1,(2*self.nElX + 2*self.nElY)+1):
            Tess[i-1]=Tb(boundaryCoordinates[:, i-1])
        self.essentialTemperatures[np.array(self.essentialNodes)-1] = (Tess[:(2*self.nElX + 2*self.nElY)])[self.boundaryType]
        self.naturalBoundaries = np.zeros((self.nEl, 4),dtype=bool)
        globNatNodes = self.boundaryNodes[np.array(nat_nodes)-1]
        globNatNodes = (np.array(globNatNodes)-1).tolist()

        for i in range(1,len(globNatNodes)+1):
            natElem = np.where(globNatNodes[i-1] == self.globalNodeNumber)

            if len(natElem[0])==2:
                elem = [natElem[0][1],natElem[0][0]]
            else:
                elem = [natElem[0]]
            if globNatNodes[i-1]==0:
                
                self.naturalBoundaries[0, 0] = True
                self.naturalBoundaries[0, 3] = True
            elif globNatNodes[i-1]==self.nElX:
               
                self.naturalBoundaries[elem, 0] = True
                self.naturalBoundaries[elem, 1] = True
            elif globNatNodes[i-1] == (self.nElX + 1)*(self.nElY + 1)-1:
                
                self.naturalBoundaries[elem, 1] = True
                self.naturalBoundaries[elem, 2] = True
            elif globNatNodes[i-1] == (self.nElX + 1)*(self.nElY):
                
                self.naturalBoundaries[elem, 2] = True
                self.naturalBoundaries[elem, 3] = True
            elif globNatNodes[i-1] > 0 and globNatNodes[i-1] < self.nElX:
                
                self.naturalBoundaries[elem[0], 0] = True
                self.naturalBoundaries[elem[1], 0] = True
            elif np.mod(globNatNodes[i-1]+1, self.nElX + 1) == 0:
                
                self.naturalBoundaries[elem[0], 1] = True
                self.naturalBoundaries[elem[1], 1] = True
            elif globNatNodes[i-1] > (self.nElX + 1)*(self.nElY):
                
                self.naturalBoundaries[elem[0], 2] = True
                self.naturalBoundaries[elem[1], 2] = True
            elif np.mod(globNatNodes[i-1]+1, self.nElX + 1) == 1:
                
                self.naturalBoundaries[elem[0], 3] = True
                self.naturalBoundaries[elem[1], 3] = True

            
        self = self.setFluxForce(qb)
        self.f_tot = self.fh + self.fs
        self = self.setNodalCoordinates()
        self = self.setBvec()
        self = self.get_essential_node_in_element()
        self = self.get_glob_natural_force()

        return self        
    
    def shrink(self):
        
        self.lc = []
        self.gridX = []
        self.gridY = []
        self.lElX = []
        self.lElY = []
        self.cum_lElX = []
        self.cum_lElY = []
        self.compute_grad = []
        self.boundaryNodes = []
        self.naturalNodes = []
        self.boundaryElements = []
        self.naturalBoundaries = []
        self.boundaryType = []
        self.lx = []
        self.ly = []
        self.AEl = []
        self.essentialBoundary = []
        self.LocalNode = []
        self.fs = []
        self.fh = []
            
        return self
    
class shapeInterp():
    def __init__(self):
        pass
    def shapeFunctionValues2(self,coarseMesh, x):
            
            row = sum(coarseMesh.cum_lElY < x[1])
            if row == 0:
                row = 1
            
            col = sum(coarseMesh.cum_lElX < x[0])
            if col == 0:
                col = 1
            
            E = (row - 1)*(coarseMesh.nElX) + col-1
            
            N = [0]*4
            N[0] = (1/coarseMesh.AEl[E])* \
                (x[0] - coarseMesh.lc[E, 1, 0])*(x[1] - coarseMesh.lc[E, 3, 1])
            N[1] = -(1/coarseMesh.AEl[E])* \
                (x[0] - coarseMesh.lc[E, 0, 0])*(x[1] - coarseMesh.lc[E, 3, 1])               
            N[2] = (1/coarseMesh.AEl[E])* \
                (x[0] - coarseMesh.lc[E, 0, 0])*(x[1] - coarseMesh.lc[E, 0, 1])
            N[3] = -(1/coarseMesh.AEl[E])* \
                (x[0] - coarseMesh.lc[E, 1, 0])*(x[1] - coarseMesh.lc[E, 0, 1])
            return N, E
    
    def shapeInterp1(self,coarseMesh,x_fine):
        nVertices = x_fine.shape[0]
        tic = time.time()
        R = np.zeros(4*nVertices)
        C = np.zeros(4*nVertices)
        Nvec = np.zeros(4*nVertices)
        mh = 0
        jwl = 4
        
        for r in range(nVertices):
            
            x = [x_fine[r, 0], x_fine[r, 1]]            
            N, E = self.shapeFunctionValues2(coarseMesh,x)        
            
            c = coarseMesh.globalNodeNumber[E, :]
            R[range(mh,jwl)] = r
            C[range(mh,jwl)] = c
            Nvec[range(mh,jwl)] = N
            mh += 4
            jwl += 4
        W = sp.csr_matrix((Nvec, (R.astype(int), C.astype(int))))
        W_assembly_time = time.time() - tic
        print('W_assembly_time=',W_assembly_time)
        return W