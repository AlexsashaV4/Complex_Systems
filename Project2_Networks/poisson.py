import matplotlib.pyplot as plt
import math

def poisson_expression(m,r):
    if m<r:
        return 0
    return r**(m-r)*math.e**(-r)/math.factorial(m-r)

r=5
WS_degree_sequence=[]
ER_degree_sequence=[]
for m in range(0,21):
    WS_degree_sequence.append(poisson_expression(m,r))
    ER_degree_sequence.append(poisson_expression(m+r,r))
                              
#fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
##ax0 = fig.add_subplot(axgrid[0:3, :])
#ax1 = fig.add_subplot(axgrid[3:, :2])

fig, [ax1, ax0] = plt.subplots(2, 1, sharex = True)
  
ax1.hist(range(0,21),range(0,22),weights=WS_degree_sequence)
plt.xlim([0, 20])
 #   ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Expected degree histogram for Watts-Strogatz")
ax1.set_xlabel("Degree")
ax1.set_ylabel("# of Nodes")

ax0.hist(range(0,21),range(0,22),weights=ER_degree_sequence)
plt.xlim([0, 20])
 #   ax1.plot(degree_sequence, "b-", marker="o")
ax0.set_title("Expected degree histogram for Erdos-Renii")
ax0.set_xlabel("Degree")
ax0.set_ylabel("# of Nodes")
fig.tight_layout()
plt.show()
