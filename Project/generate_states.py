import pandas as pd

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
    for j in range(steps):
        for k in range(steps):
            for m in range(steps):
                for l in range(steps):
                    for n in range(steps):
                        states_0=-stateRanges[0]+2*i*stateRanges[0]/(steps-1)
                        states_1=-stateRanges[1]+2*j*stateRanges[1]/(steps-1)
                        states_2=-stateRanges[2]+2*k*stateRanges[2]/(steps-1)
                        states_3=-stateRanges[3]+2*m*stateRanges[3]/(steps-1)
                        states_4=-stateRanges[4]+2*l*stateRanges[4]/(steps-1)
                        states_5=-stateRanges[5]+2*n*stateRanges[5]/(steps-1)
                        states.append([states_0, states_1, states_2, states_3, states_4, states_5])

df = pd.DataFrame.from_records(states)
df.to_csv("/opt/evorobotpy/lib/environment_states.csv",header=False)