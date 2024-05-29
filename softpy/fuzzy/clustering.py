from sklearn.metrics import pairwise_distances
import numpy as np
import copy
import functools
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClusterMixin
from .fuzzyset import DiscreteFuzzySet
from .fuzzyset import IntuitionisticFuzzySet

class FuzzyCMeans(BaseEstimator,ClusterMixin):
  '''
  Implements the fuzzy c-means algorithm. The interface is compatible with the scikit-learn library. It allows to set the number of clusters,
  a tolerance degree (for avoiding errors in numerical operations), the number of iterations before termination, the clustering metric as well
  as the fuzzifier degree.
  '''
  def __init__(self, n_clusters=2, epsilon=0.001, iters=100, random_state=None, metric='euclidean', fuzzifier=2):
    self.n_clusters = n_clusters
    self.epsilon = epsilon
    self.iters = iters
    self.random_state = random_state
    self.metric = metric
    self.fuzzifier = fuzzifier

  def fit(self, X, y=None):
    '''
    Applies clustering to the given data, by iteratively partially assigning instances to clusters and then recomputing the clusters' centroids.
    '''
    self.centroids = resample(X, replace=False, n_samples=self.n_clusters, random_state=self.random_state)
    self.cluster_assignments = np.zeros((X.shape[0], self.n_clusters))
    for it in range(self.iters):
      dists = pairwise_distances(X, self.centroids, metric=self.metric)

      self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
      
      for k in range(self.n_clusters):
        self.centroids[k] = np.sum(self.cluster_assignments[:,k][:,np.newaxis]**self.fuzzifier*X, axis=0)/(np.sum(self.cluster_assignments[:,k]**self.fuzzifier))

    self.fitted = True
    return self

  def predict(self, X):
    ''' 
    For each given instance returns the cluster with maximum membership degree. The fit method must have been called before executing this method.
    '''
    if not self.fitted:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    #self.cluster_assignments = self.cluster_assignments/np.sum(self.cluster_assignments, axis=1)[:, np.newaxis]
    return np.argmax(self.cluster_assignments, axis=1)
  
  def predict_proba(self, X):
    ''' 
    For each given instance returns the membership degrees to the computed clusters. The fit method must have been called before executing this method.
    '''
    if not self.fitted:
      raise RuntimeError("Estimator must be fitted")
    dists = pairwise_distances(X, self.centroids, metric=self.metric)
    self.cluster_assignments = (dists+self.epsilon)**(2/(1-self.fuzzifier))/(np.sum((dists+self.epsilon)**(2/(1-self.fuzzifier)), axis=1)[:,np.newaxis])
    fuzzy_sets = []
    for cl in range(self.cluster_assignments.shape[1]):
      fuzzy_sets.append(DiscreteFuzzySet(list(range(X.shape[0])), self.cluster_assignments[:,cl]))
    return np.array(fuzzy_sets)

  def fit_predict(self, X, y=None):
    self.fit(X,y)
    return self.predict(X)



class IntuitionisticFuzzyCMeans(BaseEstimator,ClusterMixin):
    '''
    Implements the Intuitionistic Fuzzy C-Means algorithm. The interface is compatible with the scikit-learn library. It allows to set 
    the number of clusters, a tolerance degree, the number of iterations before termination, and the fuzzifier degree.
    '''
    def __init__(self, n_clusters=2, epsilon=0.001, m=2, iters=100):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.m = m
        self.iters = iters
    
    def fit(self, Z, y=None, **kwargs):
        '''
        self.Xw :               the array containing the weights of every element in the universe X (defaults to an array of all 1s)
        self.V :                array of centroids;
        self.universe :         array containing all the elements in the universe X;
                                    for all z in Z, z.items is an improper subest of X
        self.w :                2d array of weights, specifically, the weight of each IFS in Z to each centroid;
                                    each line of self.w is used to calculate the weighted average of all the sets in Z 
                                    and assign the result to the corresponding centroid
        self.u :                the membership matrix of each set in Z to each centroid V;
        '''
        self.universe = functools.reduce(np.union1d, (z.items for z in Z))
        Xw = kwargs.get('Xw')
        self.Xw = np.array(Xw if Xw != None else [1]*len(self.universe))
        self.w = np.zeros((self.n_clusters, Z.size))
        #self.V = np.array([self.weighted_avg_set(i) for i in range(self.n_clusters)])
        self.V = np.random.choice(Z, self.n_clusters)
        self.u = np.zeros((self.n_clusters, Z.size))
        self.is_fitted_ = True
        return self


    def predict(self, Z):
        avg_iter_dist = 1
        curr_iter = 0

        while(avg_iter_dist > self.epsilon and curr_iter < self.iters):
            curr_iter += 1
            prev_V = copy.deepcopy(self.V)

            #COMPUTING THE MEMBERSHIP VALUES [STEP 2]
            for j in range(Z.size):
                '''
                for each set in Z, computes the distances between that set Z_j and all the centroids.
                if a centroid V_r such that dist(Z_j, V_r) == 0 is found, then the membership value of Z_j in respect to
                V_r is set to 1, and all the membership values of Z_j in respect to all the other centroids is set to 0
                '''
                dists_to_vr = self.distance(Z[j], self.V)
                r = np.where(dists_to_vr == 0)[0]
                if len(r) > 0:
                    r = r[0]
                    self.u[:,j] = 0
                    self.u[r,j] = 1
                    continue
                '''
                if there's no centroid whose distance to Z_j is equal to 0, the membership value of Z_j 
                in respect to each centroid is computed 
                '''
                for i in range(self.n_clusters): 
                    dist_to_vi = self.distance(Z[j], self.V[i])
                    powers = np.power(dist_to_vi / dists_to_vr, 2/(self.m-1))
                    sum = np.sum(powers)
                    self.u[i,j] = 1/sum
        
            #COMPUTING THE NEW WEIGHTS FOR EACH CENTROID [STEP 3]
            for i in range(self.n_clusters):
                sum = np.sum(self.u[i])
                for l in range(Z.size):
                    self.w[i,l] = self.u[i,l]/sum
                self.V[i] = self.weighted_avg_set(i, Z)

            #CHECKING THE DISTANCE TO THE PREVIOUS ITERATION [STEP 4]
            dists_prev_inter = 0
            for i in range(self.n_clusters):
                dists_prev_inter += self.distance(prev_V[i], self.V[i])
            avg_iter_dist = dists_prev_inter/self.n_clusters

        return self.u


    def fit_predict(self, Z, y=None, **kwargs):
        if(not isinstance(Z, np.ndarray)):
            Z = np.array(Z)
        self.fit(Z, **kwargs)
        return self.predict(Z)


    def weighted_avg_set(self, i, Z) -> IntuitionisticFuzzySet:
        membs = []
        for x in self.universe:
            memb_x = [0, 0]
            for j in range(Z.size):
                m = Z[j](x)
                memb_x[0] += self.w[i,j] * m[0]
                memb_x[1] += self.w[i,j] * m[1]
            membs.append(memb_x)

        return IntuitionisticFuzzySet(self.universe,membs,True)
    

    def distance(self, A, B):
        if isinstance(B, np.ndarray):
            return np.array([self.distance(A,b) for b in B])
        
        sum = 0
        for i, x in enumerate(self.universe):
            x_a = A(x)
            x_b = B(x)
            u = np.power(x_a[0] - x_b[0], 2)
            v = np.power(x_a[1] - x_b[1], 2)
            p = np.power((1-x_a[0]-x_a[1]) - (1-x_b[0]-x_b[1]), 2)
            sum += self.Xw[i]*(u+v+p)

        return np.sqrt(sum/2)



class IntuitionisticFuzzyDBSCAN(BaseEstimator,ClusterMixin):
    '''
    Implements the Intuitionistic Fuzzy DBSCAN algorithm. The interface is compatible with the scikit-learn library. It allows to set 
    the radiuses for the inner and the outer neighborhoods, the minimum number of points to be in the inner and the outer neighborhoods
    to consider the point a core point, and the fuzzy treshold.
    '''
    def __init__(self, eps1, eps2, minPts1, minPts2, alpha):
        '''
        self.eps1 :     epsilon1, the radius for the inner neighborhood 
        self.eps2 :     epsilon2, the radius for the outer neighborhood
        self.minPts1 :  the minimum number of points to be in the inner neighborhood of a point P
                        for that point to be considered a core point
        self.minPts2 :  the minimum number of points to be in the outer neighborhood of a point P
                        for that point to be considered a core point
        self.alpha :    the fuzzy threashold used to determin wheter a point 
        '''
        self.eps1 = eps1
        self.eps2 = eps2
        self.minPts1 = minPts1
        self.minPts2 = minPts2
        self.alpha = alpha
    

    def fit(self, A, y=None, **kwargs):
        '''
        self.universe : the array containing all the elements in the universe X
        self.w :        the array containing the weights of every element in the universe X (defaults to an array of all 1s)
        self.n :        the length of the data set A
        self.C :        the matrix containing the assignment for each data point in the data set A (the columns) to each cluster (the rows)
        self.noise :    an array of length n denoting whether or not the data point at the corresponding index is considered Noise or not
        self.visited :  an array of length n denoting whether or not the data point at the corresponding index has already been visited or not
        self.intfcore : an array containing all the intuitionistic fuzzy core points
        self.dist :     the distance matrix of the set (only the upper triangle is calcluated to reduce complexity)
        '''
        self.universe = functools.reduce(np.union1d, (a.items for a in A))
        w = kwargs.get('w')
        self.w = np.array(w if w != None else [1]*len(self.universe))
        self.n = len(A)
        self.C = []
        self.noise = np.array([False]*self.n)
        self.visited = np.array([False]*self.n)
        self.intfcore = []
        self.dist = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.dist[i][j] = self.distance(A[i], A[j])
                self.dist[j][i] = self.dist[i][j]
            
        return self


    def predict(self, A):
        for i in range(len(A)):     #for each not visited point p in A
            if not self.visited[i]:
                self.visited[i] = True
                N1, N2 = self.getNeighbors(i, A)    #get its neighbors
                u, v = self.corePoint(len(N1), len(N2))     #calculate its membership and non-membership degrees
                if (u < v) or (u - v < self.alpha):  #checks whether p is an IF core point
                    self.noise[i] = True    #if it's not, mark as noise
                else:
                    self.intfcore.append((i,u,v))   
                    self.C.append(self.expCluster(i, N1, A))    #if it is, create a cluster starting from P

        return np.array(self.C)


    def fit_predict(self, A, y=None, **kwargs):
        if(not isinstance(A, np.ndarray)):
            A = np.array(A)
        self.fit(A, **kwargs)
        return self.predict(A)


    def expCluster(self, o, N, A):
        c = [0]*self.n  #instatiate a cluster, at first, no point is assigned to it
        c[o] = 1    #assign the origin point o to the new cluster
        for i in N: 
            if not self.visited[i]:     #for each non visited point p in N (initially, the inner neighborhood of o)
                self.visited[i] = True
                c[i] = 1        #add p to the cluster
                N1, N2 = self.getNeighbors(i, A)    
                u, v = self.corePoint(len(N1), len(N2))     
                if (u-v) >= self.alpha: #check whether p is dense enough
                    N.extend(n for n in N1 if n not in N)   #if it is, extend the cluster to p's neighborhood
        return c


    def distance(self, A, B): #euclidean distance between two IFSs
        l = [ self.w[i] *
            (round(np.power(ax[0] - bx[0], 2),4) +
            round(np.power(ax[1] - bx[1], 2),4) +
            round(np.power((1 - ax[0] - ax[1]) - (1 - bx[0] - bx[1]), 2),4))
            for i, x in enumerate(self.universe) for ax, bx in [(A(x), B(x))] ]
        return round(np.sqrt((1/2)*sum(l)),4)


    def unitDensity(self, eps, Nop):
        return round(Nop/(np.pi * eps**2),4)


    def corePoint(self, Nop1, Nop2):
        if Nop1 >= self.minPts1:
            if Nop2 >= self.minPts2:
                u = 1
                v = 0
            else:
                u = (1 - (self.unitDensity(self.eps1, Nop1) / self.unitDensity(self.eps2, Nop2))) * (Nop2 / self.minPts2)
                v = (self.minPts2-Nop2)/self.minPts2
        else:
            if Nop2 >= self.minPts2:
                u = Nop1 / self.minPts1
                v = ((self.unitDensity(self.eps1, self.minPts1) - self.unitDensity(self.eps2, self.minPts2))/self.unitDensity(self.eps1, self.minPts1)) * (1 - (Nop1/self.minPts1))
            else:
                u = 0
                v = 1

        return u, v


    def getNeighbors(self, i, A):
        N1 = []
        N2 = []
        for j in range(self.n): #get all the points j such that are is within the give radiuses to the point i
            if j != i:
                if self.dist[i][j] <= self.eps1:
                    N1.append(j)
                if self.dist[i][j] <= self.eps2:
                    N2.append(j)
        return N1, N2



class IntuitionisticFuzzyMinimumSpanningTree(BaseEstimator,ClusterMixin):
    '''
    Implements the Intuitionistic Fuzzy MAST algorithm. The interface is compatible with the scikit-learn library. It allows to set 
    the lamba threshold to cut the MST to perform the final clustering, the alpha, beta and gamma weights for the Zhang distance
    measure, and the distance method: Zhang, Hamming or Euclidean.
    '''
    def __init__(self, lamb=None, alpha_beta_gamma=[round(1/3,4),round(1/3,4),round(1/3,4)], method=0):
        '''
        self.lamb :     the lambda value to cut the MST to perform the clustering 
        slef.abg :      the alpha_beta_gamma vector for the Zhang distance measure (defaults to [1/3, 1/3, 1/3])
        self.method :   specifies the method used to calculate the distance between to IFSs:
                                - 0 (default) means Zhang's Method
                                - 1 means Hamming distance
                                - 2 means Euclidean distance 
        '''
        if method > 2:
            raise ValueError("method can be either 0 (Zhang), 1 (Hamming) or 2 (Euclidean), is %s" % str(method))
        if len(alpha_beta_gamma) != 3:
            raise ValueError("alpha_beta_gamma must be a list of 3 values, alpha, beta and gamma in order, is a list of %s values" % str(len(alpha_beta_gamma)))
        if method>0 and not (np.issubdtype(type(lamb), np.number)):
            raise TypeError("Lambda should be a number if the selected method is Hamming (1) or Euclidean (2)")
        if method==0 and not (isinstance(lamb, tuple) or lamb==None):
            raise TypeError("Lambda should be a tuple if the selected method is Zhang (0, default)")

        self.lamb = lamb
        self.abg = alpha_beta_gamma
        self.method = method
    
    
    def fit(self, A, y=None, **kwargs):
        '''
        self.universe :     array containing all the elements in the universe X
        self.w :            the array containing the weights of every element in the universe X (defaults to an array of all 1s)
        self.n :            the length of the dataset A
        self.E :            the list that contains the edges that connect each node (IFS) with their relative weight (distance)       
        self.T :            the list that will contain the edges of the MST of the graph G = (A,E)
        self.sets :         a list that will be used to implement the sets for the Kruskal MST algorithm        
        self.rank :         a list that will be used to implement the union fuction in Kruskal MST algorithm
        self.assign :       the list that will contain the cluster assignement for each IFS:
                            for n = 5, self.assign = [0,0,1,3,2,1] means that:
                                - A[0] and A[1] are assigned to cluster 0
                                - A[2] and A[5] are assigned to cluster 1
                                - A[3] is assigend to cluster 3
                                - A[4] is assigned to cluster 2
        self.u :            the return value of predit(), matrix of size c x n where c is the number of clusters;
                            each cell [i][j] contains 1 if the IFS i is assigned to the cluster i, 0 otherwhise
        '''
        self.universe = functools.reduce(np.union1d, (a.items for a in A))
        w = kwargs.get('w')
        self.w = np.array(w if w != None else [1]*len(self.universe))
        self.n = len(A)
        self.E = []
        self.T = []
        self.sets = list(range(self.n))
        self.rank = [0]*self.n
        self.assign = list(range(self.n))
        self.u = None

        #self.E contains the edges of the undirected, complete graph G = (A, E)
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.E.append([self.distance(A[i], A[j]), [i,j]])
                
        if(self.method == 0):
            self.E.sort(key=functools.cmp_to_key(self.compare_zhang_distancess))
        else:
            self.E.sort()
            
        return self


    def predict(self, A):
        #Kruskal MST
        i = 0
        for e in self.E:
            if self.findset(e[1][0]) != self.findset(e[1][1]):
                self.T.append(e) 
                self.union(e[1][0], e[1][1])
                if ++i == self.n-1: #the MST algorithm can terminate when the number of edges in the MST is = n-1
                    break

        #cutting the edges of the MST whose weight > lamba
        if self.method == 0:    #Zhang distances
            if self.lamb == None:   #if no lamba was provided, the default value is calculated
                self.lamb = tuple(np.mean([e[0] for e in self.T], axis=0))
            cut_T = [edge[1] for edge in self.T if self.XuYagerComparison(edge[0], self.lamb) < 1]
        else:                   #Euclidean or Hamming distances
            if self.lamb == None:   
                self.lamb = np.mean(np.array([e[0] for e in self.T])) #the default lambda value is the mean of all the weights of the edges of the MST
            cut_T = [edge[1] for edge in self.T if edge[0] <= self.lamb]

        #computes the sets that form different connected components in the lamba-cut MST using the same method of Kruskal's algorithm
        self.sets = list(range(self.n))
        self.rank = [0]*self.n
        for e in cut_T:
            if self.findset(e[0]) != self.findset(e[1]):
                self.union(e[0], e[1])
        
        #to each node (IFS) assigns the index of its parent set (connected component), which is intepreted as the cluster
        self.sets = np.array(self.sets)
        self.sets = self.sets - self.sets.min()
        for i in range(self.n):
            self.assign[i] = self.findset(i)
        
        self.assign = np.array(self.assign)
        uniq = np.unique(self.assign)
        self.u = np.zeros((len(uniq), self.n))
        
        for i, cluster in enumerate(uniq):
            self.u[i] = (self.assign == cluster).astype(float)

        return self.u


    def fit_predict(self, A, y=None, **kwargs):
        if(not isinstance(A, np.ndarray)):
            A = np.array(A)
        self.fit(A, **kwargs)
        return self.predict(A)


    def distance(self, A, B):
        if self.method == 0: #Zhang
            l = [ self.w[i] *
                (self.abg[0] * np.power(ax[0] - bx[0], 2) +
                self.abg[1] * np.power(ax[1] - bx[1], 2) +
                self.abg[2] * np.power((1 - ax[0] - ax[1]) - (1 - bx[0] - bx[1]), 2))
                for i, x in enumerate(self.universe) for ax, bx in [(A(x), B(x))] ]
            return (round(np.sqrt(min(l)), 4), 1-round(np.sqrt(max(l)), 4))

        #Euclidean (p=2) or Hamming (p=1)
        p = self.method
        l = [ self.w[i] *
            (round(np.power(ax[0] - bx[0], p),4) +
            round(np.power(ax[1] - bx[1], p),4) +
            round(np.power((1 - ax[0] - ax[1]) - (1 - bx[0] - bx[1]), p),4))
            for i, x in enumerate(self.universe) for ax, bx in [(A(x), B(x))] ]
        return round(np.power(((1/2)*sum(l)),1/p),4)


    def XuYagerComparison(self, a,b):
        s = (a[0]-a[1],b[0]-b[1])
        h = (a[0]+a[1],b[0]+b[1])
        if not np.isclose(s[0], s[1], rtol=1e-05, atol=1e-08, equal_nan=False):
            return 1 if (s[0] > s[1]) else -1
        if not np.isclose(h[0], h[1], rtol=1e-05, atol=1e-08, equal_nan=False):
            return 1 if (h[0] > h[1]) else -1
        return 0


    def compare_zhang_distancess(self, a,b):
            return self.XuYagerComparison(a[0],b[0])


    def findset(self, i):
        while self.sets[i] == i:
            return i
        return self.findset(self.sets[i])


    def union(self, i, j):  
        x = self.findset(i)
        y = self.findset(j)
        if self.rank[x] < self.rank[y]:
            self.sets[x] = y
        elif self.rank[x] > self.rank[y]:
            self.sets[y] = x
        else:
            self.sets[y] = x
            self.rank[x] += 1
