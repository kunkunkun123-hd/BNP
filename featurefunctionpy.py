import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import ndimage
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cityblock
import networkx as nx
import scipy.io as scio

def edge_weightc(u, v):
    x1, y1 = u
    x2, y2 = v
    return cityblock((x1, y1), (x2, y2))

def edge_weightq(u, v):
    x1, y1 = u
    x2, y2 = v
    if np.abs(x1-x2)>np.abs(y1-y2):
        d=np.abs(x1-x2)+(np.sqrt(2)-1)*np.abs(y1-y2)
    else:
        d=(np.sqrt(2)-1)*np.abs(x1-x2)+np.abs(y1-y2)
    return d


def ddtu(matrix,type):
    G = nx.Graph()
    if type=='cityblock':
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    if i+1<matrix.shape[0] and matrix[i+1, j]!=0:
                        G.add_edge((i, j), (i+1, j), weight=edge_weightc((i, j), (i+1, j)) if i < matrix.shape[0]-1 else np.inf)
                    if j+1<matrix.shape[1] and matrix[i, j+1]!=0:
                        G.add_edge((i, j), (i, j+1), weight=edge_weightc((i, j), (i, j+1)) if j < matrix.shape[1]-1 else np.inf)
                    if i-1>0 and matrix[i-1, j] != 0:
                        G.add_edge((i, j), (i-1, j), weight=edge_weightc((i, j), (i-1, j)) if i > 0 else np.inf)
                    if j-1>0 and matrix[i, j-1] != 0:
                        G.add_edge((i, j), (i, j-1), weight=edge_weightc((i, j), (i, j-1)) if j > 0 else np.inf)

    elif type=='quasi-euclidean':
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    if i+1<matrix.shape[0] and matrix[i+1, j]!=0:
                        G.add_edge((i, j), (i+1, j), weight=edge_weightq((i, j), (i+1, j)))
                    if j+1<matrix.shape[1] and matrix[i, j+1]!=0:
                        G.add_edge((i, j), (i, j+1), weight=edge_weightq((i, j), (i, j+1)))
                    if i-1>0 and matrix[i-1, j] != 0:
                        G.add_edge((i, j), (i-1, j), weight=edge_weightq((i, j), (i-1, j)))
                    if j-1>0 and matrix[i, j-1] != 0:
                        G.add_edge((i, j), (i, j-1), weight=edge_weightq((i, j), (i, j-1)))

                    if j-1>0 and i-1>0 and matrix[i-1, j-1] != 0:
                        G.add_edge((i, j), (i-1, j-1), weight=edge_weightq((i, j), (i-1, j-1)) if (i>0 and j > 0) else np.inf)

                    if j+1<matrix.shape[1] and i-1>0 and matrix[i-1, j+1] != 0:
                        G.add_edge((i, j), (i-1, j+1), weight=edge_weightq((i, j), (i-1, j+1))) 

                    if j-1>0 and i+1<matrix.shape[0] and matrix[i+1, j-1] != 0:
                        G.add_edge((i, j), (i+1, j-1), weight=edge_weightq((i, j), (i+1, j-1))) 

                    if j+1<matrix.shape[1] and i+1<matrix.shape[0] and matrix[i+1, j+1] != 0:
                        G.add_edge((i, j), (i+1, j+1), weight=edge_weightq((i, j), (i+1, j+1)))   
    return G

def zajl(matrix,G,C,R):
    D=matrix.copy()
    E=D>0
    mh,jwl=E.shape
    for i in range(mh):
        for j in range(jwl):
            lxq=[]
            if E[i,j]:
                    try:
                        for k in range(len(C)):
                            target=(R[k]-1,C[k]-1)
                            lxq.append(nx.shortest_path_length(G, source=(i,j),target=target, weight='weight'))
                        D[i,j]=float(min(np.array(lxq)))
                    except:
                        D[i,j]=np.inf
            else:
                D[i,j]=np.nan
    return D

def volumeFractionCircExclusions(diskCenters, diskRadii, gridRF):

    A = np.zeros(gridRF.nCells)
    A0 = A.copy()
    circle_surfaces = (np.pi * diskRadii ** 2).T
    n = 1

    for cll in gridRF.cells:
        A[n-1] = cll.surface
        A0[n-1] = A[n-1]
        circles_in_n = cll.inside(diskCenters)
        A[n-1] = A0[n-1] - np.sum(circle_surfaces[circles_in_n])
        n += 1

    porefrac = A / A0
    porefrac[porefrac <= 0] = np.finfo(float).eps  
    return porefrac

def chordLengthDensity(diskCenters, diskRadii, gridRF, distance):

    diskRadii=diskRadii.T
    meanRadii = np.zeros(gridRF.nCells)
    meanSqRadii = np.zeros(gridRF.nCells)
    A = np.zeros(gridRF.nCells)
    A0 = A.copy()

    n = 1
    for cll in gridRF.cells:
            radii_in_n = diskRadii[cll.inside(diskCenters)]
            meanRadii[n-1] = np.mean(radii_in_n)
            meanSqRadii[n-1] = np.mean(radii_in_n**2)
            A0[n-1] = cll.surface
            A[n-1] = cll.surface - np.pi*np.sum(radii_in_n**2)
            n = n + 1

    porefrac = A/A0
    porefrac[porefrac <= 0] = np.finfo(float).eps  
    exclfrac = 1 - porefrac

    lc = .5*np.pi*(meanSqRadii/meanRadii)*(porefrac/exclfrac)
    if np.any(~np.isfinite(lc)):
        print('Setting mean chord length of cells with no inclusion to 0.')
        lc[np.isnan(lc)] = np.sqrt(A[np.isnan(lc)])  

    cld = (1./lc) * np.exp(-(distance/lc))
    return cld

def interfacePerVolume(diskCenters, diskRadii, gridRF):
    
    A0 = np.zeros(gridRF.nCells)
    interface_area = np.zeros(gridRF.nCells)
    circle_circumferences = 2 * np.pi * diskRadii.T
    n = 0
    
    for cll in gridRF.cells:
            A0[n] = cll.surface
            circles_in_n = cll.inside(diskCenters)
            interface_area[n] = np.sum(circle_circumferences[circles_in_n])
            n += 1
    
    relativeInterfaceArea = interface_area / A0
    
    return relativeInterfaceArea

def squareWellPotential(distances, d):

    V = -sum(distances[distances < d])
    return V

def lennardJonesPotential(distances, d):

    V = np.sum((d/distances)**12 - (d/distances)**6)
    return V

def diskDistance(disk_centers, disk_radii, grid_RF, property, p_norm, pot_param=[]):
    disk_radii=disk_radii.T
    dist_quantity = np.zeros((grid_RF.nCells))
    edg_max = dist_quantity.copy()
    n = 1
    for cll in grid_RF.cells:
        exception_flag = False
        centers = disk_centers[cll.inside(disk_centers), :]
        radii = disk_radii[cll.inside(disk_centers)]
        distances = np.zeros((int(radii.size*(radii.size-1)/2)))
        ind = 1
        if radii.size > 1:
            for i in range(1,radii.size+1):
                for j in range(i+1, radii.size+1):
                    if p_norm == 'edge2edge':
                        distances[ind-1] = np.linalg.norm(centers[i-1, :] - centers[j-1, :])
                        distances[ind-1] = distances[ind-1] - radii[i-1] - radii[j-1]
                    else:
                        distances[ind-1] = np.linalg.norm(centers[i-1, :] - centers[j-1, :], p_norm)                   
                    ind += 1
           
        else:
            distances = 0
            for edg in range(1,len(cll.edges)+1):
                try: 
                    if cll.edges[edg-1].length > edg_max[n-1]:
                        edg_max[n-1] = cll.edges[edg-1].length
                except:
                    print('cell with deleted edge')
            exception_flag = True    
        if property == 'mean':
                dist_quantity[n-1] = np.mean(distances)
        elif property == 'max':
                dist_quantity[n-1] = np.max(distances)
        elif property == 'min':
                dist_quantity[n-1] = np.min(distances)
        elif property == 'std':
                dist_quantity[n-1] = np.std(distances)
        elif property == 'var':
                dist_quantity[n-1] = np.var(distances)
        elif property == 'squareWellPot':
                dist_quantity[n-1] = squareWellPotential(distances, pot_param)
        elif property == 'lennardJones':
                dist_quantity[n-1] = lennardJonesPotential(distances, pot_param)
        else:
            raise ValueError('Unknown distance property')
            
        if exception_flag:
            dist_quantity[n-1] = 0.5 * edg_max[n-1]
            if property == 'var':
                dist_quantity[n-1] = dist_quantity[n-1]**2
            
        n += 1
    
    return dist_quantity
    

def matrixLinealPath(diskCenters, diskRadii, gridRF, distance):
    diskRadii=diskRadii.T
    meanRadii = np.zeros(gridRF.nCells)
    meanSqRadii = np.zeros(gridRF.nCells)
    A = np.zeros(gridRF.nCells)
    A0 = A.copy()

    n = 1
    for cll in gridRF.cells:
            radii_in_n = diskRadii[cll.inside(diskCenters)]
            meanRadii[n-1] = np.mean(radii_in_n)
            meanSqRadii[n-1] = np.mean(radii_in_n**2)
            A0[n-1] = cll.surface
            A[n-1] = cll.surface - np.sum(np.pi*radii_in_n**2)
            n += 1

    porefrac = A/A0
    porefrac[porefrac <= 0] = np.finfo(float).eps 

    L = porefrac*np.exp(-(2*distance*(1 - porefrac)*meanRadii)/(np.pi*porefrac*meanSqRadii))
    L[np.isnan(L)] = 1 

    return L

def voidNearestSurfaceExclusion(diskCenters, diskRadii, gridRF, distance):
    
    diskRadii=diskRadii.T
    meanRadii = np.zeros(gridRF.nCells)
    meanSqRadii = np.zeros(gridRF.nCells)
    A = np.zeros(gridRF.nCells)
    A0 = np.zeros(gridRF.nCells)

    n = 1
    for cll in gridRF.cells:
            inside = cll.inside(diskCenters)
            radii_in_n = diskRadii[inside]
            meanRadii[n-1] = np.mean(radii_in_n)
            meanSqRadii[n-1] = np.mean(np.square(radii_in_n))
            A0[n-1] = cll.surface
            A[n-1] = cll.surface - np.pi*np.sum(np.square(radii_in_n))
            n += 1

    porefrac = A/A0
    porefrac[porefrac <= 0] = np.finfo(float).eps  

    exclfrac = 1 - porefrac
    S = np.square(meanRadii)/meanSqRadii
    a_0 = (1 + exclfrac*(S - 1))/(np.square(porefrac))
    a_1 = 1/porefrac

    x = distance/(2*meanRadii)
    F = np.exp(-4*exclfrac*S*(a_0*np.square(x) + a_1*x))
    e_v=porefrac*F
    if any(~np.isfinite(e_v)):
        e_v[~np.isfinite(e_v)] = 1
    if any(~np.isfinite(F)):
        F[~np.isfinite(F)] = 1
    h_v = 2*((exclfrac*S)/(meanRadii))*(2*a_0*x + a_1)*e_v
    P = 2*((exclfrac*S)/(meanRadii))*(2*a_0*x + a_1)*F
    if any(~np.isfinite(h_v)):
        h_v[~np.isfinite(h_v)] = 0
    if any(~np.isfinite(P)):
        P[~np.isfinite(P)] = 0
    return e_v, h_v, P, F

def meanChordLength(diskCenters, diskRadii, gridRF):
    
    diskRadii=diskRadii.T
    meanRadii = np.zeros(gridRF.nCells)
    meanSqRadii = np.zeros(gridRF.nCells)
    edg_max = meanRadii.copy()
    A = np.zeros(gridRF.nCells)
    A0 = A.copy()
    n = 1
    for cll in gridRF.cells:
        radii_in_n = diskRadii[cll.inside(diskCenters)]
        meanRadii[n-1] = np.mean(radii_in_n)
        meanSqRadii[n-1] = np.mean(radii_in_n**2)
        A0[n-1] = cll.surface
        A[n-1] = cll.surface - np.pi*np.sum(radii_in_n**2)
        for edg in range(1,len(cll.edges)+1):
            if cll.edges[edg-1].length > edg_max[n-1]:
                edg_max[n-1] = cll.edges[edg-1].length   
        n = n + 1
        
    porefrac = A/A0
    porefrac[porefrac <= 0] = np.finfo(float).eps
    exclfrac = 1 - porefrac
    
    lc = .5*np.pi*(meanSqRadii/meanRadii)*(porefrac/exclfrac)
    
    lc[lc > edg_max] = edg_max[lc > edg_max]
    lc[~np.isfinite(lc)] = edg_max[~np.isfinite(lc)]
    return lc

def momentPerVolume(diskCenters, diskRadii, gridRF, moment):

    diskRadii=diskRadii.T
    A0 = np.zeros(gridRF.nCells)
    sum_radii_moments = A0.copy()
    radii_samples = diskRadii**moment
    n = 1
    for cll in gridRF.cells:
            A0[n-1] = cll.surface
            circles_in_n = cll.inside(diskCenters)
            sum_radii_moments[n-1] = np.sum(radii_samples[circles_in_n])
            n = n + 1

    relativeInterfaceArea = sum_radii_moments/A0
    return relativeInterfaceArea

def linPathLengthScale(diskCenters, diskRadii, grd, distances):

    L = np.zeros((grd.nCells, len(distances)))
    for i in range(1,len(distances)+1):
        L[:, i-1] = matrixLinealPath(diskCenters, diskRadii, grd, distances[i-1])
    
    L = L + np.finfo(float).eps
    out = np.zeros(grd.nCells)
    for k in range(1,grd.nCells+1):
        f,_ = curve_fit(lambda x, a,b: a*np.exp(b*x), distances, L[k-1, :])
        out[k-1] = f[1]
    return out

def distanceTransform(lambdaMat, distMeasure, phaseInversion, meanVarMaxMin):

    if ~phaseInversion:
        lambdaMat = np.logical_not(lambdaMat)
    if distMeasure =='euclidean':
        dist = ndimage.distance_transform_edt(lambdaMat)
    elif distMeasure =='cityblock':
        dist = ndimage.distance_transform_cdt(lambdaMat,metric=distMeasure) 
    elif distMeasure =='chessboard':
        dist = ndimage.distance_transform_cdt(lambdaMat,metric=distMeasure) 
    else:
         TypeError('unknown distMeasure')
    
    if np.any(np.isinf(dist)):
        if distMeasure == 'cityblock':
            dist[np.isinf(dist)] = dist.shape[0] + dist.shape[1]
        elif distMeasure == 'chessboard':
            dist[np.isinf(dist)] = max(dist.shape[0], dist.shape[1])
        else:
            dist[np.isinf(dist)] = np.linalg.norm(dist.shape)

    if meanVarMaxMin == 'mean':
        out = np.mean(dist)
    elif meanVarMaxMin == 'var':
        out = np.var(dist)
    elif meanVarMaxMin == 'max':
        out = np.amax(dist)
    elif meanVarMaxMin == 'min':
        print('Min dist is usually 0 for every cell. Feature should not be used.')
        out = np.amin(dist)
    else:
        raise ValueError('Mean or variance of lambda bubble property?')

    return out

def gaussLinFilt(lambda_mat, muGaussFilt=None, sigmaGaussFiltFactor=10):
    
    rows, cols = lambda_mat.shape
    
    if muGaussFilt is None or np.isnan(muGaussFilt).any():
        muGaussFilt = [(rows + 1) / 2, (cols + 1) / 2]
        
    sigmaGaussFilt = sigmaGaussFiltFactor * np.array([rows, cols])
    
    x, y = np.meshgrid(np.arange(1, rows+1), np.arange(1, cols+1))
    xy = np.column_stack((x.flatten(), y.flatten()))
    
    w = multivariate_normal.pdf(xy, mean=muGaussFilt, cov=sigmaGaussFilt)
    w = w / np.sum(w)
    w = w.reshape(rows, cols)
    
    debug = False
    if debug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(w)
        plt.draw()
        plt.pause(1)
    
    X = np.sum(np.multiply(w, lambda_mat))
    return X

def isingEnergy(lambdakMat):

    nr, nc = lambdakMat.shape

    E = 0
    for c in range(1,nc+1):
        for r in range(1,nr+1):
            if(r > 1):
                if(lambdakMat[r-1, c-1] == lambdakMat[r - 2, c-1]):
                    E = E + 1
                else:
                    E = E - 1
                
            else:

                if(lambdakMat[0, c-1] == lambdakMat[nr-1, c-1]):
                    E = E + 1
                else:
                    E = E - 1
            if(r < nr):
                if(lambdakMat[r-1, c-1] == lambdakMat[r, c-1]):
                    E = E + 1
                else:
                    E = E - 1
            else:

                if(lambdakMat[nr-1, c-1] == lambdakMat[0, c-1]):
                    E = E + 1
                else:
                    E = E - 1
            if(c > 1):
                if(lambdakMat[r-1, c-1] == lambdakMat[r-1, c-2]):
                    E = E + 1
                else:
                    E = E - 1
            else:

                if(lambdakMat[r-1, 0] == lambdakMat[r-1, nc-1]):
                    E = E + 1
                else:
                    E = E - 1
            if(c < nc):
                if(lambdakMat[r-1, c-1] == lambdakMat[r-1, c]):
                    E = E + 1
                else:
                    E = E - 1
            else:

                if(lambdakMat[r-1, nc-1] == lambdakMat[r-1, 0]):
                    E = E + 1
                else:
                    E = E - 1
    return E

def shortestPath(lambdak, dir, resolution=256, distMeasure='cityblock'):
    
    lambdak = ~lambdak
    shortestPath = np.inf
    if dir == 'x':
        
        left = lambdak[:, 0]
        right = lambdak[:, -1]
        jwl=lambdak
        jwl=jwl.astype(int)
        jwl=jwl.astype(float)

        G=ddtu(jwl,distMeasure)

        if any(left) and any(right):
           
            leftIndex = round(lambdak.shape[0] / 2)
            incr = 0
            while not np.isfinite(shortestPath) and 1 < leftIndex < lambdak.shape[0]:
                leftIndex += incr
                incr = -(incr + 1)
                jwl=lambdak
                jwl=jwl.astype(int)
                jwl=jwl.astype(float)

                geo = zajl(jwl,G,[1], [int(leftIndex)])
                if any(np.isfinite(geo[:,-1])):
                    shortestPathTemp = np.min(geo[:,-1])
                    if shortestPathTemp < shortestPath:
                        shortestPath = shortestPathTemp
                        if not np.isfinite(shortestPath):
                            print(leftIndex)
                            print(dir)
                            print(geo)
                            raise ValueError('Zero path length of connected path')
            if not np.isfinite(shortestPath):
                shortestPath = np.pi * left.shape[0]
    elif dir == 'y':
        
        top = lambdak[0, :]
        bottom = lambdak[-1, :]
        if any(top) and any(bottom):
            
            topIndex = round(lambdak.shape[1] / 2)
            incr = 0
            while not np.isfinite(shortestPath) and 1 < topIndex < lambdak.shape[1]:
                topIndex += incr
                incr = -(incr + 1)
                jwl=lambdak
                jwl=jwl.astype(int)
                jwl=jwl.astype(float)
                geo = zajl(jwl, [1], [int(topIndex)], distMeasure)
                if any(np.isfinite(geo[-1, :])):
                    shortestPathTemp = min(geo[-1, :])
                if(shortestPathTemp < shortestPath):
                    shortestPath = shortestPathTemp
                    if not np.isfinite(shortestPath):
                        print(topIndex)
                        print(dir)
                        print(geo)
                        raise ValueError('Zero path length of connected path')

            if not np.isfinite(shortestPath):
               shortestPath = np.pi*top.shape[1] 
    else:
        raise ValueError('which direction?')
    shortestPath = (shortestPath + 1)/resolution
    return shortestPath