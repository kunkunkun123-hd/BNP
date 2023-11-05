import numpy as np
import scipy.io as scio

class Mesh:
    def __init__(self):
        self.vertices = []  
                             
        self.edges = []     
                             
        self.cells = []     
                              
        self.nVertices = 0  
        
        self.nEdges = 0     
        self.nCells = 0

    def create_vertex(self, coordinates):
        
        vtx = Vertex(coordinates)
        self.vertices.append(vtx)
        self.nVertices += 1
        return vtx
    
    def create_edge(self,vertex1, vertex2):
        edg = Edge(vertex1, vertex2)
        self.edges.append(edg)
        vertex1.add_edge(edg)
        vertex2.add_edge(edg)
        self.nEdges += 1
        return edg 
    
    def create_cell(self, vertices, edges):
            
        new_cell = Cell(vertices, edges)
        self.cells.append(new_cell)
    
        for vertex in vertices:
            vertex.add_cell(new_cell)
    
        for edge in edges:
            edge.add_cell(new_cell)
    
        self.nCells += 1




class Vertex:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.cells = []  
        self.edges = [] 

    def add_cell(self, cell):
        self.cells.append(cell)

    def add_edge(self, edge):
        self.edges.append(edge)


class Edge:
    def __init__(self, vertex1, vertex2):
        self.vertices = [vertex1, vertex2]
        vertexc=[vertex1.coordinates[i]- vertex2.coordinates[i] for i in range(len(vertex1.coordinates))]
        self.length = np.linalg.norm(vertexc)
        self.cells = []
    
    def add_cell(self, cell):
        self.cells.append(cell)

class Cell:

    def __init__(self, vertices, edges):
        
        self.vertices = vertices
        self.edges = edges
        self.type = 'rectangle'

        centroid1 = sum(v.coordinates[0] for v in self.vertices)/len(self.vertices)
        centroid2 = sum(v.coordinates[1] for v in self.vertices)/len(self.vertices)
        self.centroid = [centroid1, centroid2]

        self.compute_surface()

    def compute_surface(self):
        
        if self.type == 'rectangle':
            self.surface = self.edges[0].length*self.edges[1].length

    def inside(self,x):
        eps = np.finfo(float).eps
        try:
            isin=(x[:,0] > self.vertices[0].coordinates[0] - eps) & \
            (x[:,0] <= self.vertices[2].coordinates[0] + eps) & \
            (x[:,1] > self.vertices[0].coordinates[1] - eps) & \
            (x[:,1] <= self.vertices[2].coordinates[1] + eps)
        except:
        
            isin=(x[0] > self.vertices[0].coordinates[0] - eps) & \
            (x[0] <= self.vertices[2].coordinates[0] + eps) & \
            (x[1] > self.vertices[0].coordinates[1] - eps) & \
            (x[1] <= self.vertices[2].coordinates[1] + eps)
        
        return isin

        
class RectangularMesh(Mesh):

    def __init__(self,gridX, gridY=[]):
        super().__init__()
        if len(gridY)==0:
            gridY = gridX
            
        x_coord = np.cumsum(np.append(0,gridX))
        y_coord = np.cumsum(np.append(0,gridY))
            
        self.x_coord = x_coord
        self.y_coord = y_coord

        for y in y_coord:
            for x in x_coord:
                super().create_vertex([x, y])
        try:
            nx = len(gridX) + 1
            ny = len(gridY) + 1
        except:
            nx=gridX+ 1
            ny=gridY+ 1
        for y in range(ny):
            for x in range(nx):
                if self.vertices[x+y*nx].coordinates[0] > 0:
                    super().create_edge(self.vertices[x + y*nx - 1], self.vertices[x + y*nx])
        
        for y in range(ny):
            for x in range(nx):
                if self.vertices[x + y*nx].coordinates[1] > 0:
                    super().create_edge(self.vertices[x + (y - 1)*nx], self.vertices[x + y*nx])

        nx = nx - 1
        ny = ny - 1
        n = 0  
        for y in range(1, ny + 1):
            for x in range(0, nx):
                vtx = [self.vertices[x + (y - 1)*(nx + 1)], 
                       self.vertices[x + (y - 1)*(nx + 1) + 1],
                       self.vertices[x + y*(nx + 1) + 1], 
                       self.vertices[x + y*(nx + 1)]]
                edg= [self.edges[n], self.edges[nx*(ny + 1) + n + y],
                       self.edges[n + nx], 
                       self.edges[nx*(ny + 1) + n + y - 1]]
                super().create_cell(vtx, edg)
                n = n + 1
    
    def map2fine(self,toMesh):
        
        map=np.zeros((toMesh.nCells,self.nCells))
        n=1
        for cll in self.cells:
            try: 
                m=1
                try:
                    for to_cll in toMesh.cells:
                        isin=cll.inside(to_cll.centroid)
                        if isin:
                            map[m-1,n-1]=1
                        m=m+1
                except:
                    print('to_cll is not valid')  
            except:
                print('cll in not valid')
            n=n+1
        return map

    def indexIndicator(self,resolution=256):

        x = np.linspace(0,1,resolution + 1)
        x=0.5*(x[1:]+x[0:-1])
        xx,yy = np.meshgrid(x,x)
        m = 1
        ind=[]
        nrow=[]
        ncol=[]
        for cll in self.cells:
            mh1=np.vstack(xx.flatten('F'))
            mh2=np.vstack(yy.flatten('F'))
            mh3=np.hstack((mh1,mh2))
            ind.append(np.reshape(cll.inside(mh3),(resolution,resolution),order='F'))
            ncol.append(max(np.sum(ind[m-1],axis=1)))
            nrow.append(max(np.sum(ind[m-1],axis=0)))
            m = m + 1
        return ind,nrow,ncol    