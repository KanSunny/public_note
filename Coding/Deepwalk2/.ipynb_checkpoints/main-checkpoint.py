
# coding: utf-8

# In[7]:


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter # 命令行解析模块
# from deepwalk import graph
# from deepwalk import walks
# from deepwalk.wordvec import Word2Vec
import random

def process(args):
#     if args.format == "adjlist":
#         G = graph.load_adjacencylist(args.input, undirected=args.undirected)
#     elif args.format == "edgelist":
#         G = graph.load_adjacencylist(args.input, undirected=args.undirected)
#     else:
    raise Exception("unknown file format: '%s'. valid formats: 'adjlist', 'edgelist'" % args.format)

def main():
    parser: ArgumentParser = ArgumentParser("deepwalk", formatter_class=ArgumentDefaultsHelpFormatter,
                                          conflict_handler="resolve")# 创建解析器对象
    parser.add_argument('--format', default='adjlist', choices=['adjlist', 'edgelist'])
    parser.add_argument('--number-walks', default=10, type=int)
    parser.add_argument('--input', nargs='?', required=True)
    #parser.add_argument('--output', required=True)
    parser.add_argument('--representation-size', default=64, type=int, help='number of latent dimensions to learn for each code.')
    parser.add_argument('--seed', default=0, type=int)#?
    parser.add_argument('--undirected', default=True, type=bool, help='treat graph as undirected.')#？
    parser.add_argument('--vertex-freq-degree', default=False, action='store_true', 
                       help='using the vertex degree to estimate the frequency of nodes '
                            'in the random walk. this option is faster than '
                            'calculating the vocabulary')# ？
    parser.add_argument('--walk-length', default=40, type=int)
    parser.add_argument('--window-size', default=5, type=int, help='window size of skigram model.')
    
    args = parser.parse_args("--input ./example_graphs/karate.adjlist"
                             "--output ./output"
                             "--representation-size 2".split())
    
    print(args)
    process(args)
    
if __name__ == "__main__":
    main()

