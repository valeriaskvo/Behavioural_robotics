import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

def get_action(observation,W1,W2,b1,b2):
    # convert the observation array into a matrix with 1 column and ninputs rows
    observation.resize(ninputs,1)
    Z1 = np.dot(W1, observation) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)
    if (isinstance(env.action_space, gym.spaces.box.Box)):
        action = A2
    else:
        action = np.argmax(A2)
    return action

def initialize_weights(ninputs,nhiddens,noutputs,pvariance):
    W1=np.random.randn(nhiddens,ninputs) * pvariance
    W2=np.random.randn(noutputs, nhiddens) * pvariance
    b1=np.zeros(shape=(nhiddens, 1))
    b2=np.zeros(shape=(noutputs, 1))
    return W1, W2, b1, b2

def update_weights(best_in_pop,ninputs,nhiddens,noutputs,ppvariance):
    l_pop=[]
    for i in range(len(best_in_pop)):
        W1, W2, b1, b2=best_in_pop[i]
        dW1, dW2, db1, db2=initialize_weights(ninputs,nhiddens,noutputs,ppvariance)
        l_pop.append([W1+dW1,W2+dW2,b1+db1,b2+db2])
        dW1, dW2, db1, db2=initialize_weights(ninputs,nhiddens,noutputs,ppvariance)
        l_pop.append([W1+dW1,W2+dW2,b1+db1,b2+db2])
    return l_pop

def run_steps(env,W1,W2,b1,b2,steps,vis_flag=0,t=10):
    env.reset()
    eval=0
    action=0
    for _ in range(steps):
        observation, reward, done, info = env.step(action)
        eval+=reward
        action=get_action(observation,W1,W2,b1,b2)
        if vis_flag==1:
            env.render()
            sleep(t/steps)
    return eval

def find_best_in_pop(pop_size,l_pop,env,W1,W2,b1,b2,steps):
    rewards=np.zeros(pop_size)
    for i in range(pop_size):
        W1, W2, b1, b2=l_pop[i]
        rewards[i]=run_steps(env,W1,W2,b1,b2,steps)
    
    best_l_pop=np.argsort(rewards)
    best_in_pop=[]
    for i in best_l_pop[int(pop_size/2):]:
        best_in_pop.append(l_pop[i])
    
    eval=sum(rewards[best_l_pop[int(pop_size/2):]])/(steps*pop_size/2)

    return best_in_pop, eval

env=gym.make('CartPole-v0')
steps=200
epochs=50

pop_size=10

pvariance=0.1
ppvariance=0.005
nhiddens=5
ninputs = env.observation_space.shape[0]

if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n

l_pop=[]
evolution=[]

for i in range(pop_size):
    W1, W2, b1, b2=initialize_weights(ninputs,nhiddens,noutputs,pvariance)
    l_pop.append([W1,W2,b1,b2])

evolution.append(l_pop)

results=np.zeros((epochs,2))

for i in range(epochs):
    best_in_pop, eval=find_best_in_pop(pop_size,l_pop,env,W1,W2,b1,b2,steps)
    l_pop=update_weights(best_in_pop,ninputs,nhiddens,noutputs,ppvariance)
    evolution.append(l_pop)
    results[i,:]=[i+1,eval]
    print(i+1,eval)

W1, W2, b1, b2=best_in_pop[-1]
run_steps(env,W1,W2,b1,b2,steps,vis_flag=1,t=10)

plt.plot(results[:,0],results[:,1])
plt.title('Evaluation neural network by summary reward')
plt.xlabel('Epochs')
plt.ylabel('Summary reward')
plt.xlim([0,epochs+1])
plt.ylim([0,1.1])
plt.grid('on')
plt.show()