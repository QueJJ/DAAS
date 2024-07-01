import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import math
import matplotlib.pyplot as plt
import random as rand
import networkx as nx

# scaled sla and bound
sla = 10
GBound = 60
ResourceLimit = 42

class DAAS(object):
    def __init__(self, meshgrid, contexts, G, paths):
        # beta term
        self.Bg = GBound
        self.Rg = 1
        self.p = 0.05

        # penalty initialization
        self.Q = 1

        # grid_term
        self.contexts = contexts
        self.N = len(meshgrid)
        self.meshgrid = np.array(meshgrid)
        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.actions = self.meshgrid.shape[1]

        # graph info
        self.graph = G
        self.paths = paths
        self.topological_order = list(nx.topological_sort(self.graph))
        
        # for each node, we construct a GP, except node 0
        self.mugList = []
        self.sigmagList = []
        self.gtList = []

        for node in self.topological_order:
            parents = list(G.predecessors(node))
            inputSpace = 0
            if parents != []:
                contextsRange = self.contexts[parents[0]]
                inputSpace = self.actions * len(contextsRange)
            else:
                inputSpace = self.actions
            self.mugList.append(np.array([0 for _ in range(inputSpace)]))
            self.sigmagList.append(np.array([1 for _ in range(inputSpace)]))
            self.gtList.append(np.array([0 for _ in range(inputSpace)]))

        # round counter
        self.round = 0

        # decision list, resources list and latency list
        self.index = 0
        self.XList = []
        self.latencysList = []
        for i in range(self.N):
            self.XList.append([])
            self.latencysList.append([])

    def getE2eLatencyP2C(self, x):
        # record the position of each node under certain context and the context
        tmpPosition = [0] * self.N
        tmpContext = [0] * self.N
        # get the context for each node and the latency
        for node in self.topological_order:
            parents = list(self.graph.predecessors(node))
            pp = x[node] - 1
            if parents != []:
                parentsContext = tmpContext[parents[0]]
                parentsContextsRange = self.contexts[parents[0]]
                contextsRange = self.contexts[node]
                context_idx = np.where(parentsContextsRange <= parentsContext)[0][0]

                lowerBound = context_idx * self.actions
                upperBound = (context_idx+1) * self.actions
                eval_context = np.clip(self.gtList[node][lowerBound:upperBound][pp], min(contextsRange), max(contextsRange))
                tmpContext[node] = eval_context
                tmpPosition[node] = context_idx * self.actions + pp
            else:
                contextsRange = self.contexts[node]
                eval_context = np.clip(self.gtList[node][pp], min(contextsRange), max(contextsRange))
                tmpContext[node] = eval_context
                tmpPosition[node] = pp

        return max(tmpContext), tmpPosition, parentsContext

    def decision(self): 
        bestConfig = None
        bestPosition = [0] * self.N

        tmpConfig = []
        tmpValue = []
        tmpPosition = []
        tmpParentContext = []

        for i in self.X_grid:
            e2eLatency, position, parentcontext = self.getE2eLatencyP2C(i)
            tmpConfig.append(i)
            tmpValue.append(ResourceLimit - np.sum(i) - self.Q * max(e2eLatency - sla, 0))
            tmpPosition.append(position)
            tmpParentContext.append(parentcontext)
        idx = np.argmax(tmpValue)
        bestConfig = tmpConfig[idx]
        bestPosition = tmpPosition[idx]

        self.round += 1
        for i in range(self.N):
            self.XList[i].append(np.array([bestPosition[i]+1]))
        return bestConfig

    def update(self, resources, segments_latency):
        cp_latency = max(segments_latency)
        for i in range(self.N):
            self.latencysList[i].append(np.clip(segments_latency[i], min(self.contexts[i]), max(self.contexts[i])))
        
        #update beta
        gamma = 1*math.log(self.round+1)
        betag = self.Bg + self.Rg * np.sqrt(2 * (gamma + 1 + np.log(2 / self.p)))

        # update penalty
        eta = 1*math.sqrt(self.round)
        self.Q = max(self.Q + max(cp_latency - sla, 0), eta)

        # GP update
        for i in range(self.N):
            grid = np.arange(1, 1+len(self.gtList[i])).reshape(-1, 1)
            gpg = GaussianProcessRegressor()
            
            gpg.fit(self.XList[i], self.latencysList[i])
            self.mugList[i], self.sigmagList[i] = gpg.predict(grid, return_std=True)
            self.gtList[i] = self.mugList[i] - betag * self.sigmagList[i]
            