import pandas as pd

def generate_seed(set_state,states):
    seed=0
    for i in range(len(set_state)):
        state=set_state[i]
        for j in range(len(states)):
            if abs(state-states[j][i])<0.0000001:
                seed+=(j+1)*10**i
                break
    return seed

env_states=pd.read_csv("/opt/evorobotpy/xdpole/environment_states_ranges.csv",header=None)
env_states=env_states.values.tolist()

labels=['Iteration','State 1','State 2','State 3','State 4','State 5','State 6','Seed', 'gen', 'eval', 'bestfit', 'bestgfit', 'centroid', 'bestsam', 'avg', 'weightsize', 'runtime']

stateRanges_0 = 1.944
stateRanges_1 = 1.215
stateRanges_2 = 0.10472
stateRanges_3 = 0.135088
stateRanges_4 = stateRanges_2
stateRanges_5 = stateRanges_3
stateRanges=[stateRanges_0, stateRanges_1, stateRanges_2, stateRanges_3, stateRanges_4, stateRanges_5]

steps=5
states=[]
for i in range(steps):
    state=[]
    for j in range(len(stateRanges)):
        state.append(-stateRanges[j]+2*i*stateRanges[j]/(steps-1))
    states.append(state)


states_dataset=[]
for state in env_states:
    set_state=state[1:]
    seed=generate_seed(set_state,states)
    try:
        f=open('/opt/evorobotpy/xdpole/one_init_state/S'+str(seed)+'.fit','r')
        for line in f:
            fit_res=line.split()
        result=fit_res[1:len(fit_res):2]
        for item in result:
            state.append(item)
        states_dataset.append(state)
        
    except FileNotFoundError:
        print('Ooooooops!')
        continue

df = pd.DataFrame.from_records(states_dataset)
df.to_csv("/opt/evorobotpy/xdpole/enfironment_results_data.csv",header=labels,index=False)