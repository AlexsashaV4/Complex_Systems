import snap
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import alive_progress as ap
from scipy.stats import ks_2samp
import pyarma as pa
import math
import powerlaw
colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
sns.set_theme()

# Similarity between two probability distributions

"""KL Divergence(P|Q) it is a measure of how one probability distribution is different from a second, reference probability distribution."""
def KL_div(p_probs, q_probs):    
    p_probs = np.asarray(p_probs)
    q_probs = np.asarray(q_probs)
    KL_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(KL_div)
"""Jensen Shannon Divergence"""
def JS_Div(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (KL_div(p, m) + KL_div(q, m)) / 2
"""Kolmogorov-Smirnov Distance"""
def KS_Dist(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    return np.max(np.abs(np.cumsum(p) - np.cumsum(q)))
"""Hill's estimator """
#or from scipy.stats import ks_2samp
#ks_2samp(p, q)
# implement the Hill's estimator for the degree distribution assuming that is scale free
def Hill_estimator(data):
    """
    Returns the Hill Estimators for some 1D data set.
    """    
    # sort data in such way that the smallest value is first and the largest value comes last:
    Y = np.sort(data)
    n = len(Y)

    Hill_est = np.zeros(n-1)

    for k in range(0, n-1):    # k = 0,...,n-2
        summ = 0

        for i in range(0,k+1):   # i = 0, ..., k
            summ += np.log(Y[n-1-i]) - np.log(Y[n-2-k])
        
        Hill_est[k] = (1 / (k+1)) * summ      # add 1 to k because of Python syntax
  
    kappa = 1 / Hill_est
    return kappa

# chi^2 test
def chi2_test(data, degree_sequence):
    k = np.arange(1, max(data)+1)
    n = len(data)
    degree_sequence = np.asarray(degree_sequence)
    degree_sequence = degree_sequence/np.sum(degree_sequence)
    degree_sequence = degree_sequence*n
    chi2 = np.sum((data - degree_sequence)**2/degree_sequence)
    return chi2
# 
# implement the maximum likelihood estimator for the degree distribution assuming it is poisson distributed
def max_likelihood_poisson(data):
    r = np.mean(data)
    degree_sequence = [np.exp(-r)*r**k/math.factorial(k) for k in range(0, max(data)+1)]
    return r, degree_sequence
# maximum likelihood estimator for the degree distribution assuming it is power law distributed (alpha-1)*k^(-alpha)
def max_likelihood_powerlaw(data):
    n = len(data)
    alpha = 1 + n/(np.sum(np.log(data)))
    return alpha
   

"""
INPUT: A string containing the name of the file to be loaded (.txt file)

This function takes a graph and calculates the cluster coefficient and the degree distribution.
If the graph is scale free, it also calculates the parameter alpha of the graph, which is the slope of the degree distribution
in a log-log plot. 
The function also plots the degree distribution.
    """
def f(title):
    for i in title:
        print("=============================================")
        print("Extracting the Graph from:", i)
        graph=snap.LoadEdgeList(snap.TUNGraph, i, 0, 1)
        #graph.DelZeroDegNodes()
        # Graph size 
    #     print("Number of nodes:", graph.GetNodes())
    #     print("Number of edges:", graph.GetEdges())
    #     G = nx.read_edgelist(i, nodetype=int)
    #     # exclude the isolated nodes from the graph
    #     G.remove_nodes_from(list(nx.isolates(G)))
    #     A = nx.adjacency_matrix(G)
    #     #         # find the number of edges between neighbors of a vertex using the third power of the adjacency matrix Cluster_i = sum_{j != k} A_{ij} A_{jk} A_{ki}
    #     print("Self loops:", A.diagonal().sum())
    #     # average Cluster coefficient
    #     cf = nx.average_clustering(G)
    #     print("Cluster coefficient with nx:", cf)
    #     # From networkx we export the cluster coefficient because the definition is the same as the one used in our report
    #     cf2 = nx.clustering(G)
    #     cf2 = np.array(list(cf2.items()))
    #     print("Cluster coefficient with nx mean:", np.mean(cf2[cf2<=1]))
    #     #Thanks to the SNAP lybrary we calculate the cluster coefficient 
    #     #In theory this should be calculated by using the adjacency matrix**3 but the matrix is too large for this
    #     cf = graph.GetClustCf()
    #     print("Cluster coefficient:", cf)
    #     # Get a list of the nodes in the graph and degree of each node
    #     nodes = [node.GetId() for node in graph.Nodes()]
    #     degr = [node.GetInDeg() for node in graph.Nodes()]
    #     # Check if in and out degrees are the same
    #     degr = np.array(degr)
    #     print ("Same in and out degrees?", np.array_equal(degr, [node.GetOutDeg() for node in graph.Nodes()]))
    #     # Find the degree of a vertex by using the adjacency matrix
    #     # we can write it using the third power of the adjacency matrix C_i = sum_{j != k} A_{ij} A_{jk} A_{ki}
    #     #{of edges between neighbors of i}

    #     #find the number of edges between neighbors of i using the third power of the adjacency matrix
    #     v_deg = np.array(A.sum(axis=0))[0] # vector of degrees of the nodes

    #     #print("v_deg",v_deg.shape , "max", max(v_deg), "min", min(v_deg))
    #     #compare v_deg with degrees to check if they are the same
    #    # print("Same?", np.array_equal(v_deg, A2.diagonal()))
    #     # Calculate the clustering coefficient from the adjacency matrix excluding the nodes with degree 0 or 1
    #     # We do this because the clustering coefficient is not defined for these nodes 
    #     # Use the third power of the adjacency matrix to calculate the clustering coefficient
    #     not_zero = np.where(degr>1)[0]
    #     #A3diag = (A3.diagonal())[not_zero]
    #     # count how many times the diagonal is 0 and where v_deg is 0
    #     A2 = (A.dot(A))
    #     A3 = (A2.dot(A))
    #     A3diag = (A3.diagonal())
    #     cf2 = np.zeros(len(A3diag))
    #     for i in range(len(A3diag)):
    #         if v_deg[i] > 1:
    #             cf2[i] = A3diag[i]/(v_deg[i]*(v_deg[i]-1))
    #     print("Cluster coefficient from the adjacency matrix:", np.mean(cf2))
    #     cf2alt= np.sum(A3diag)/(np.sum(v_deg[not_zero]*(v_deg[not_zero]-1)))
    #     print("Cluster coefficient from the adjacency matrix alternative:", cf2alt)

        #We calculate the degree distribution of the graph by using the SNAP library
        CntV = graph.GetOutDegCnt()
        #This is done in a log-log plot
        #The graph is scale free iff it is a straight line
        degrees=[]
        counts=[]
        # We only consider the datapoints for k not too large.
        # The reason for this is, that the graph is not large enough...
        # ...to give reliable data for very large k.
        # Therefore, we cutoff when we get count 0 for the first time.
        # This happens when a values of the degrees is skipped.
        # Also, this helps because we don't get 0 values of which the logarithm would be -infinity
        for p in CntV:
            if len(degrees)>1: # We need at least two values to compare
                if degrees[-1]>degrees[-2]+1:
                    break
            if p.GetVal1()!=0 and p.GetVal2()!=0:
                degrees.append(p.GetVal1())
                counts.append(p.GetVal2())
        log_degrees=[np.log(x) for x in degrees ]
        number_of_vertices=sum(counts)
        log_counts=[np.log(x) for x in counts]
        # print the shape of v_deg

        #make the plot decent looking
        
        norm_vert = [a/number_of_vertices for a in counts]
        print("np.unique(norm_vert)", np.unique(norm_vert))
        #We delete values which are the logarithm of infinity
    
        # Plot the poisson distribution that fits the degree distribution
        # This is only done for the roadNet-CA.txt graph
        if i==title[0]:
            plt.style.use("seaborn")
            plt.plot( degrees,[a/number_of_vertices for a in counts], linestyle='', marker='o', markersize=4.0, label= "Degree distribution " + i, color=colors[0])
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # We calculate the average degree of the graph
            average_degree=2*graph.CntUniqUndirEdges()/graph.GetNodes()
            # We calculate the poisson distribution
            poisson=[np.exp(-average_degree)*average_degree**k/np.math.factorial(k) for k in degrees]
            # We plot the poisson distribution
            plt.plot( degrees,poisson, linestyle='', marker='o', markersize=4.0, label= "Estimated Poisson distribution", color=colors[1])
            plt.xlabel("k")
            plt.ylabel("P(k)")
            plt.title("Degree distribution of " + i + " graph")
            plt.legend(loc ='best')
            plt.savefig("degree_distribution_pois_nolog.eps")
            plt.show()
            # Calculate the KS distance between the poisson distribution and the degree distribution
            print("KS distance between the poisson distribution and the degree distribution:",KS_Dist(poisson,[a/number_of_vertices for a in counts]))
            # Calculate the JS divergence between the poisson distribution and the degree distribution
            print("JS divergence between the poisson distribution and the degree distribution:",JS_Div(poisson,[a/number_of_vertices for a in counts]))
            # Calculate the KL divergence between the poisson distribution and the degree distribution
            print("KL divergence between the poisson distribution and the degree distribution:",KL_div(poisson,[a/number_of_vertices for a in counts]))
            # Calculate the chi-squared distance between the poisson distribution and the degree distribution
            # Calculate the chi square distance between the linear regression and the degree distribution
            print("Chi square distance between the linear regression and the degree distribution:",chi2_test([a/number_of_vertices for a in counts],poisson))
        # plot the linear regression in the log-log plot with the slope alpha
        elif i==title[1]:
            plt.style.use("seaborn")
            plt.plot( degrees,norm_vert, linestyle='', marker='o', markersize=3.0, label= "degree distribution " + i)
            #ax.scatter(degrees,[a/number_of_vertices for a in counts] ,markersize = 0.5 ,label= "degree distribution " + i, color=colors[title.index(i)+1])
            # best alpha from max likelihood use the function 
            print("unique degrees:", np.unique(np.asarray(degrees)), "max degree:", max(degrees), "min degree:", min(degrees))
            # plot the maximum likelihood estimate

            # We calculate the maximum likelihood estimate
            # We use the function from the powerlaw library
            m = max_likelihood_powerlaw(degrees) 
            print("alpha from max likelihood:", m)
            plt.plot(degrees,(m-1)*degrees**(-m), label=" Power law estimation with  MLE", color = 'mediumpurple')
            # m,b = np.polyfit(np.log(degrees), np.log(norm_vert), 1)
            # print("alpha:",m)
            # plt.plot(degrees,np.exp(b)*degrees**m, label="Linear regression", color = colors[1])
            # poisson distribution
            #average_degree=2*graph.CntUniqUndirEdges()/graph.GetNodes()
            # We calculate the poisson distribution
            #poisson=[np.exp(-average_degree)*average_degree**k/np.math.factorial(k) for k in degrees]
            #plt.plot(degrees,poisson, label="Estimated Poisson distribution", color=colors[2])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel("k")
            plt.ylabel("Number of vertices with degree k")
            plt.title("Degree distribution of the graphs")
            plt.legend()
            plt.savefig("degree_distribution_scale.eps")
            plt.show()
            # Calculate the KS distance between the linear regression and the degree distribution
            print("KS distance between the linear regression and the degree distribution:",KS_Dist([(m-1)*x**(-m) for x in degrees],[a/number_of_vertices for a in counts]))
            # Calculate the JS divergence between the linear regression and the degree distribution
            print("JS divergence between the linear regression and the degree distribution:",JS_Div([(m-1)*x**(-m) for x in degrees],[a/number_of_vertices for a in counts]))
            # Calculate the KL divergence between the linear regression and the degree distribution
            print("KL divergence between the linear regression and the degree distribution:",KL_div([(m-1)*x**(-m) for x in degrees],[a/number_of_vertices for a in counts]))
            # Calculate the chi square distance between the linear regression and the degree distribution
            print("Chi square distance between the linear regression and the degree distribution:",chi2_test([a/number_of_vertices for a in counts][100:],[(m-1)*x**(-m) for x in degrees][100:]))
            # hill estimator
            # print("Hill estimator:", Hill_estimator([a/number_of_vertices for a in counts]))
            # # plot the hill estimator
            
            # plt.figure(figsize=(15, 6))
            # #plot the data points
            # h = Hill_estimator(norm_vert[4:-100])
            # plt.plot(degrees, [a/number_of_vertices for a in counts], 'o', color='black')
            # plt.plot(np.unique(degrees)[4:-101], (h)*(2**h)*np.unique(degrees)[4:-101]**(-(h+1)), 'green')
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel(r"$k$ (number of order statistics)", fontsize=18)
            # plt.ylabel("fit with (Hill Estimate)", fontsize=18)
            # plt.title("Hill Plot (right tail)", fontsize=20)
            # plt.show()
        print("The graph is scale free with parameter alpha?")
        print("number of nodes:",number_of_vertices)
        number_of_edges=graph.CntUniqUndirEdges()
        print("number of edges:", number_of_edges)
        print("average degree:",number_of_edges*2/number_of_vertices)
        # We will now calculate the average degree of the neighbors of a vertex.
        # We ignore isolated nodes
        CntV = graph.GetOutDegCnt()
        average_degrees=[]
        for NI in graph.Nodes():
            degrees=[]
            for Id in NI.GetOutEdges():
                degrees.append(graph.GetNI(Id).GetOutDeg())
            if len(degrees)>0:
                average_degrees.append(np.mean(degrees))
        the_average=np.mean(average_degrees)
        print("average degree of neighbors",the_average)
        print("=============================================")
titles=["roadNet-TX.txt","com-dblp.ungraph.txt"]
f(titles)
