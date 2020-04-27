import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import cos
from math import fabs
from sklearn.cluster import DBSCAN

def calculate_energy_init_state(state):
	GRAVITY = 9.8
	MASSCART = 1.0	
	M_1 = 0.1
	M_2 = 0.05;
	L_1 = 0.5
	L_2 = 0.25;
	I_1=1/3*M_1*L_1**2
	I_2=1/3*M_2*L_2**2
	x=state[0]
	dx=state[1]
	theta_1=state[2]
	dtheta_1=state[3]
	theta_2=state[4]
	dtheta_2=state[5]
	
	K_cart=MASSCART*dx**2/2
	P_cart=0
	
	K_pole_1=I_1*dtheta_1**2/2
	P_pole_1=M_1*GRAVITY*cos(theta_1)*L_1/2

	K_pole_2=I_2*dtheta_2**2/2
	P_pole_2=M_2*GRAVITY*cos(theta_2)*L_2/2
	
	return K_cart,P_cart,K_pole_1,P_pole_1,K_pole_2,P_pole_1, K_cart+P_cart+K_pole_1+P_pole_1+K_pole_2+P_pole_1

def calculate_the_difficulty(state):
	x=state[0]
	dx=state[1]

	theta_1=state[2]
	dtheta_1=state[3]

	theta_2=state[4]
	dtheta_2=state[5]
	
	metrix_cart=fabs(x)**2
	metrix_pole_1=fabs(theta_1)**2
	metrix_pole_2=fabs(theta_2)**2
	metrix_pole_12=fabs(theta_1-theta_2)**2

	return metrix_cart,metrix_pole_1,metrix_pole_2,metrix_pole_12, metrix_cart+metrix_pole_1+metrix_pole_2+metrix_pole_12


#THIRTY_SIX_DEGREES=0.628329
#TRACK_EDGE=2.4

name='enfironment_results_data.csv'
env_states=pd.read_csv(name,header=None)
env_states=env_states.values.tolist()
env_states=np.asarray(env_states[1:],dtype='float')
state=env_states[:,1:7]


X=np.zeros((len(state),2))
energy=np.zeros((len(state),6))
metrix=np.zeros((len(state),4))
state_i=state[0]

for i in range(len(state)):
	state_i=state[i]
	e1,e2,e3,e4,e5,e6,e_sum=calculate_energy_init_state(state_i)
	m1,m2,m3,m4,m_sum=calculate_the_difficulty(state_i)
	X[i,0]=e_sum
	X[i,1]=m_sum
	energy[i,0]=e1
	energy[i,1]=e2
	energy[i,2]=e3
	energy[i,3]=e4
	energy[i,4]=e5
	energy[i,5]=e6
	
	metrix[i,0]=m1
	metrix[i,1]=m2
	metrix[i,2]=m3
	metrix[i,3]=m4

#"""
plt.subplot(3,1,1)
plt.title('Best fit versus sqare value of pole angles in initial state')
plt.scatter(metrix[:,1],env_states[:,10])
plt.xlabel('Sqare value of angle 1')
plt.ylabel('Best fit')
plt.subplot(3,1,2)
plt.scatter(metrix[:,2],env_states[:,10])
plt.xlabel('Sqare value of angle 2')
plt.ylabel('Best fit')
plt.subplot(3,1,3)
plt.scatter(metrix[:,3],env_states[:,10])
plt.xlabel('Sqare value of difference between two angles')
plt.ylabel('Best fit')
plt.show()		
	

plt.title('Best fit versus sqare value of cart position in initial state')
plt.scatter(metrix[:,0],env_states[:,10])
plt.xlabel('Sqare value of cart position')
plt.ylabel('Best fit')
plt.show()

"""

xlabels=['Kinetic energy of cart','Potential energy of cart','Kinetic energy of pole 1','Potential energy of pole 1','Kinetic energy of pole 2','Potential energy of pole 2']

for i in range(6):
	plt.subplot(3,2,i+1)
	plt.scatter(energy[:,i],env_states[:,10])
	plt.xlabel(xlabels[i])
	plt.ylabel('Best fit')
	plt.xlim((np.min(energy[:,i])-3*np.std(energy[:,i]),np.max(energy[:,i])+3*np.std(energy[:,i])))
	print('Mean value of '+xlabels[i])
	print(np.mean(energy[:,i]))
	print('Standart deviation of '+xlabels[i])
	print(np.std(energy[:,i]))
plt.show()
"""

### Clasterization by the difficulty 
db = DBSCAN(eps=0.1, min_samples=10).fit(X)
labels = db.labels_
unique_labels = set(labels)


core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
	if k == -1:
        # Black used for noise.
		col = [0, 0, 0, 1]

	class_member_mask = (labels == k)

	xy = X[class_member_mask & core_samples_mask]
	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             	markeredgecolor='k', markersize=14)

	xy = X[class_member_mask & ~core_samples_mask]
	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)


plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xlabel('Init state energy')
plt.ylabel('Init state difficulty')
plt.show()
