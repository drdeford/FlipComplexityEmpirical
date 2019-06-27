import os
import random
import json
import geopandas as gpd
import functools
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from networkx.readwrite import json_graph
import math
import seaborn as sns
from functools import partial
import networkx as nx
import numpy as np


from gerrychain import Graph
from gerrychain import MarkovChain
from gerrychain.constraints import (Validator, single_flip_contiguous,
within_percent_of_ideal_population, UpperBound)
from gerrychain.proposals import propose_random_flip, propose_chunk_flip
from gerrychain.accept import always_accept
from gerrychain.updaters import Election,Tally,cut_edges
from gerrychain import GeographicPartition
from gerrychain.partition import Partition
from gerrychain.proposals import recom
from gerrychain.metrics import mean_median, efficiency_gap
from gerrychain.tree import recursive_tree_part





mu = 2.63815853
bases = [.1,1/mu**2,.2,1/mu,.8,1,mu,4,mu**2,10]
base=.3
pops=[.05,.1,.5,.9]
ns=25

m=50

def fixed_endpoints(partition):
    return partition.assignment[(19,0)] != partition.assignment[(20,0)] and partition.assignment[(19,39)] != partition.assignment[(20,39)]
    
    
def boundary_condition(partition):
    
    blist = partition["boundary"]
    o_part = partition.assignment[blist[0]]

    for x in blist:
        if partition.assignment[x] != o_part:
            return True
    
    return False
    
    
def boundary_slope(partition,m=m):

    a=[]
    b=[]
    c=[]
    d=[]
    e=[]
    
    for x in partition["cut_edges"]:
        if x[0][0] == 0 and x[1][0] == 0:
            a.append(x)
        elif x[0][1] == -m+1 and x[1][1] == -m+1:
            b.append(x)
        elif x[0][0] == m-1 and x[1][0] == m-1:
            c.append(x)
        elif x[0][1] == m and x[1][1] == m:
            d.append(x)
        #elif x in [((0,1),(1,0)), ((0,38),(1,39)), ((38,0),(39,1)), ((38,39),(39,38))]:
        #    e.append(x)
        #elif x in [((1,0),(0,1)), ((1,39),(0,38)), ((39,1),(38,0)), ((39,38),(38,39))]:
        #    e.append(x)
                    

    return list(set(a+b+c+d))
    

def annealing_cut_accept_backwards(partition):
    boundaries1  = {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})
    boundaries2  = {x[0] for x in partition.parent["cut_edges"]}.union({x[1] for x in partition.parent["cut_edges"]})
    
    t = partition["step_num"]
    
    
    #if t <100000:
    #    beta = 0  
    #elif t<400000:
    #    beta = (t-100000)/100000 #was 50000)/50000
    #else:
    #    beta = 3
    base = .1
    beta = 5
        
    bound = 1
    if partition.parent is not None:
        bound = (base**(beta*(-len(partition["cut_edges"])+len(partition.parent["cut_edges"]))))*(len(boundaries1)/len(boundaries2))  
        
        if not popbound(partition):
            bound = 0         
        if not single_flip_contiguous(partition):
            bound = 0             
        #bound = min(1, (how_many_seats_value(partition, col1="G17RATG",
         #col2="G17DATG")/how_many_seats_value(partition.parent, col1="G17RATG",
         #col2="G17DATG"))**2  ) #for some states/elections probably want to add 1 to denominator so you don't divide by zero


    return random.random() < bound
    
    
def go_nowhere(partition):
    return partition.flip(dict())
    
    
def slow_reversible_propose(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    #b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    flip = random.choice(list(partition["b_nodes"]))
    
    return partition.flip({flip[0]: flip[1]})

def slow_reversible_propose_bi(partition):
    """Proposes a random boundary flip from the partition in a reversible fasion
    by selecting uniformly from the (node, flip) pairs.
    Temporary version until we make an updater for this set.
    :param partition: The current partition to propose a flip from.
    :return: a proposed next `~gerrychain.Partition`
    """

    #b_nodes = {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
    #           }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

    fnode = random.choice(list(partition["b_nodes"]))
    
    return partition.flip({fnode: -1*partition.assignment[fnode]})

def geom_wait(partition):
    return int(np.random.geometric(len(list(partition["b_nodes"]))/(len(partition.graph.nodes)**(len(partition.parts))-1) ,1))-1
    

def b_nodes(partition):
    return {(x[0], partition.assignment[x[1]]) for x in partition["cut_edges"]
               }.union({(x[1], partition.assignment[x[0]]) for x in partition["cut_edges"]})

def b_nodes_bi(partition):
    return {x[0] for x in partition["cut_edges"]}.union({x[1] for x in partition["cut_edges"]})   


def uniform_accept(partition):

    bound = 0
    if popbound(partition) and single_flip_contiguous(partition) and boundary_condition(partition):
        bound = 1
        
    return random.random() < bound
        
#BUILD GRAPH



def cut_accept(partition):
      
    bound = 1
    if partition.parent is not None:
        bound = (partition["base"]**(-len(partition["cut_edges"])+len(partition.parent["cut_edges"])))#*(len(boundaries1)/len(boundaries2))
 
       

    return random.random() < bound

def step_num(partition):
    parent = partition.parent

    if not parent:
        return 0


    return parent["step_num"] + 1



fips="05"
newdir = f"./plots/States/{fips}/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")
for alignment in ['BG','COUSUB','Tract','County']:    
    for pop1 in pops:
        for base in bases:


            graph = Graph.from_json("./State_Data2/"+alignment+fips+".json")
            bnodes = [x for x in graph.nodes() if graph.node[x]["boundary_node"] ==1]

            def bnodes_p(partition):

                
                return [x for x in graph.nodes() if graph.node[x]["boundary_node"] ==1]            
            
            def new_base(partition):
                return base
    

                
            graph = Graph.from_json("./State_Data2/"+alignment+fips+".json")
            df = gpd.read_file("./State_Data2/"+alignment+fips+".shp")
            centroids = df.centroid
            c_x = centroids.x
            c_y = centroids.y
            totpop = 0
            for n in graph.nodes():
                graph.node[n]["TOTPOP"]=int(graph.node[n]["TOTPOP"])
            
                totpop += graph.node[n]["TOTPOP"]
                
            cddict = recursive_tree_part(graph,[-1,1],totpop/2,"TOTPOP", .05,1)
            
            for n in graph.nodes():            
                graph.node[n]["part_sum"]=cddict[n]
                graph.node[n]["last_flipped"]=0
                graph.node[n]["num_flips"]=0
                
            for edge in graph.edges():
                graph[edge[0]][edge[1]]['cut_times'] = 0                 
                

            
            pos = {node:(c_x[node],c_y[node]) for node in graph.nodes}   
            
                         

    
            updaters = {'population': Tally('TOTPOP', alias="population"),
                                #"boundary":bnodes_p,
                                #"slope": boundary_slope,
                                'cut_edges': cut_edges,
                                'step_num': step_num,
                                'b_nodes':b_nodes_bi,
                                'base':new_base,
                                'geom':geom_wait,
                                #"Pink-Purple": Election("Pink-Purple", {"Pink":"pink","Purple":"purple"})
                    }                  
                                
                                
    
                                
    
            #########BUILD PARTITION
    
            grid_partition = Partition(graph,assignment=cddict,updaters=updaters)
    
            #ADD CONSTRAINTS
            popbound=within_percent_of_ideal_population(grid_partition,pop1) 

            plt.figure()
            nx.draw(graph, pos = pos, node_color = [dict(grid_partition.assignment)[x] for x in graph.nodes()] ,node_size = ns,cmap = 'tab20', node_shape ='o')
            plt.savefig(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"start.png")
            plt.close()
            
            
            plt.figure()
            df["starting"] = df.index.map(dict(grid_partition.assignment))
            df.plot(column = "starting",cmap='tab20')
            plt.axis('off')
            plt.savefig(f"./plots/States/{fips}/df"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"start.png")
            plt.close()
            
            
                     
    
            #########Setup Proposal
            ideal_population = sum(grid_partition["population"].values()) / len(grid_partition)
    
            tree_proposal = partial(recom,
                                   pop_col="population",
                                   pop_target=ideal_population,
                                   epsilon=0.05,
                                   node_repeats=1
                                  )
                                  
            #######BUILD MARKOV CHAINS
    
    
            exp_chain = MarkovChain(slow_reversible_propose_bi ,Validator([single_flip_contiguous,popbound#,boundary_condition
            ]), accept = cut_accept, initial_state=grid_partition, 
            total_steps =  10000)
    
    
    
    
    
            #########Run MARKOV CHAINS
    
            rsw = []
            rmm = []
            reg = []
            rce = []
            rbn=[]
            waits= []
    
            slopes = []
            angles = []
    
            import time
    
            st = time.time()
    
    
            t=0
            for part in exp_chain:
                rce.append(len(part["cut_edges"]))
                waits.append(part["geom"])
                rbn.append(len(list(part["b_nodes"])))
                #print(part["slope"])
                
                for edge in part["cut_edges"]:
                    graph[edge[0]][edge[1]]["cut_times"] += 1
                    #print(graph[edge[0]][edge[1]]["cut_times"])


    
                
                if part.flips is not None:
                    f = list(part.flips.keys())[0]
                    graph.node[f]["part_sum"]=graph.node[f]["part_sum"]-part.assignment[f]*(t-graph.node[f]["last_flipped"])
                    graph.node[f]["last_flipped"]=t
                    graph.node[f]["num_flips"]=graph.node[f]["num_flips"]+1
    
                t+=1
                """
                plt.figure()
                nx.draw(graph, pos = pos, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='c',cmap = 'tab20')
                plt.savefig(f"./Figures/recom_{part['step_num']:02d}.png")
                plt.close()
                """
            print("finished no", st-time.time())
            with open(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"wait.txt",'w') as wfile:
                wfile.write(str(sum(waits)))

    


            for n in graph.nodes():
                if graph.node[n]["last_flipped"] == 0:
                    graph.node[n]["part_sum"]=t*part.assignment[n]
                graph.node[n]["lognum_flips"] = math.log(graph.node[n]["num_flips"] + 1) 

    
                        
                        
            #print(len(rsw[-1]))        
            #print(graph[(1,0)][(0,1)]["cut_times"])

            plt.figure()
            nx.draw(graph, pos = pos, node_color = [0 for x in graph.nodes()] ,node_size = 10, edge_color = [graph[edge[0]][edge[1]]["cut_times"] for edge in graph.edges()], node_shape ='o',cmap = 'jet',width =3)
            plt.savefig(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"edges.png")
            plt.close()

            plt.figure()
            df["ending"] = df.index.map(dict(part.assignment))
            df.plot(column = "ending",cmap='tab20')
            plt.savefig(f"./plots/States/{fips}/df"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"end.png")
            plt.axis('off')
            plt.close()
    
            plt.figure()
            nx.draw(graph, pos = pos, node_color = [dict(part.assignment)[x] for x in graph.nodes()] ,node_size = ns, node_shape ='o',cmap = 'tab20')
            plt.savefig(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"end.png")
            plt.close()
    
    

    
    
            plt.figure()
            nx.draw(graph, pos = pos, node_color = [graph.nodes[x]["part_sum"] for x in graph.nodes()] ,node_size = ns, node_shape ='o',cmap = 'jet')
            plt.savefig(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"wca.png")
            
            plt.close()
    
    
            plt.figure()
            df["partsum"] = df.index.map({x: graph.nodes[x]["part_sum"] for x in graph.nodes()})
            df.plot(column = "partsum",cmap='jet')
            plt.savefig(f"./plots/States/{fips}/df"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"wca.png")
            plt.axis('off')
            plt.close()
    
    
            plt.figure()
            plt.title("Flips")
            nx.draw(graph,pos= pos,node_color=[graph.nodes[x]["num_flips"] for x in graph.nodes()],node_size=ns,node_shape ='o',cmap="jet")
            plt.title("Flips")
            plt.savefig(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"flip.png")
            plt.close()
    
    
            plt.figure()
            df["flips"] = df.index.map({x: graph.nodes[x]["num_flips"] for x in graph.nodes()})
            df.plot(column = "flips",cmap='jet')
            plt.savefig(f"./plots/States/{fips}/df"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"flips.png")
            plt.axis('off')
            plt.close()


            plt.figure()
            plt.title("Flips")
            nx.draw(graph,pos= pos,node_color=[graph.nodes[x]["lognum_flips"] for x in graph.nodes()],node_size=ns,node_shape ='o',cmap="jet")
            plt.title("Flips")
            plt.savefig(f"./plots/States/{fips}/"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"logflip.png")
            plt.close()
    
    
            plt.figure()
            df["logflips"] = df.index.map({x: graph.nodes[x]["lognum_flips"] for x in graph.nodes()})
            df.plot(column = "logflips",cmap='jet')
            plt.savefig(f"./plots/States/{fips}/df"+str(alignment)+"B"+str(int(100*base))+"P"+str(int(100*pop1))+"logflips.png")
            plt.axis('off')
            plt.close()
