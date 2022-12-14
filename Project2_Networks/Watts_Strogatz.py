import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math

def analytical_expression(m,r,p):
    summation=0
    for n in range(min(m-r,r)+1):
        summation+=math.comb(r,n)*(1-p)**n*p**(r-n)*(r*p)**(m-r-n)*np.e**(-p*r)/math.factorial(m-r-n)
    return summation

def f(p):
    watts_strogatz = nx.watts_strogatz_graph(500,10,p)
    #watts_strogatz = nx.gnp_random_graph(100, 0.02, seed=10374196)
    
    empirical_degree_sequence = sorted((d for n, d in watts_strogatz.degree()), reverse=True)
    dmax = max(empirical_degree_sequence)
    
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    
    ax0 = fig.add_subplot(axgrid[0:3, :])
    Gcc = watts_strogatz.subgraph(sorted(nx.connected_components(watts_strogatz), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
    nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("The graph with p = "+str(p))
    ax0.set_axis_off()
    
    ax1 = fig.add_subplot(axgrid[3:, :2])
    print(empirical_degree_sequence)
    expected_degree_sequence=[]

    for x in range(5,21):
        expected_degree_sequence.append(analytical_expression(x,5,p))
    print(expected_degree_sequence)
    ax1.bar(np.arange(5,21),expected_degree_sequence,align='center',width=1,color='#1f77b4')
    plt.xlim([5, 20])
 #   ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Expected degree distribution $p_m$")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")
    plt.xticks(np.arange(5,21))
    ax1.grid()
    
    ax2 = fig.add_subplot(axgrid[3:, 2:]) 
    sequence, counts= np.unique(empirical_degree_sequence, return_counts=True)
    print('sequence:',sequence)
    print('counts:', counts)
    ax2.bar(sequence,counts/500,align='center',width=1)
    plt.xlim([5, max(sequence)])
    ax2.set_title("Empirical degree distribution $\hat{p}_m$")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")
    plt.xticks(sequence)
    ax2.grid()
    fig.tight_layout()
    #save figure in eps format
    plt.savefig("Watts_Strogatz_p_"+str(p)+".eps")
    plt.show()
    #plot the expected degree distribution and the empirical degree distribution on the same plot
    #plot the graph
    plt.figure()
    # bar plot centered on the integer values
    plt.bar(sequence,counts/500,align='center',width=1 ,color='#ff7f0e', alpha=0.8, label='Empirical')
    plt.bar(np.arange(5,21),expected_degree_sequence,align='center', color='#1f77b4', label='Expected')
    plt.title("Degree distributions")
    plt.legend()
    plt.ylabel("p(k)")
    plt.xlabel("Degree")
    plt.grid()
    plt.xlim([5, 20])
    # save figure in eps format
    plt.savefig("Watts_Strogatz_p_"+str(p)+"_combined.eps")
    plt.show()


f(0.2)
f(0.4)
f(1)