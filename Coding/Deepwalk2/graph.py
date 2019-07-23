
# coding: utf-8

# In[8]:


from collections import defaultdict # 使用dict时key不存在报错
from six import iterkeys
import random

class Graph(defaultdict):
    def __init__(self):
        super().__init__(list) # 调用父类中的变量，并为父类初始化为list
    
    def nodes(self):
        return self.keys()
    
    def adjacency_iter(self):
        return self.iters()
    
    def make_undirected(self): # 将有向图转化为无向图
        for v in self.keys(): # self.keys()是什么？
            for other in self[v]:# self[v]是什么?
                if v != other:
                    self[other].append(v)
        self.make_consistent()
        return self
    
    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k]))) # set() 函数创建一个无序不重复元素集， sort() 函数用于对原列表进行排序
            
        self.remove_self_loops()
        
        return self
    
    def remove_self_loops(self):
        for x in self:
            if x in self[x]:
                self[x].remove(x)

        return self
    
    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        G=self
        if start:
            path=[start]
        else:
            path=[rand.choice(list(G.keys()))]
            
        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break

        return [str(node) for node in path]

    
def load_adjacencylist(file_str, undirected=False, unchecked = True):
    file = open(file_str)
    if unchecked:
        adjlist = parse_adjacencylist_unchecked(file)
        G = convert_from_adjlist_unchecked(adjlist)
    else:
        adjlist = parse_adjacencylist(file)
        G = convert_from_adjlist(adjlist)

    if undirected:
        G = G.make_undirected()

    return G

def parse_adjacencylist(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend([row])

    return adjlist


def parse_adjacencylist_unchecked(f):
    adjlist = []
    for l in f:
        if l and l[0] != "#":
            adjlist.extend([[int(x) for x in l.strip().split()]])

    return adjlist


def convert_from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def convert_from_adjlist_unchecked(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node) # return 但是不会结束使用for循环可以逐次调出来

