#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   es.py runs an evolutionary expriment or post-evaluate an evolved robot/s
   type python3 es.py for help

   Requires policy.py, evoalgo.py, and salimans.py
   Also requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py   
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin 
"""


import numpy as np
import pandas as pd
import configparser
import sys
import os
import random


# global variables
scriptdirname = os.path.dirname(os.path.realpath(__file__))  # Directory of the script .py
#sys.path.insert(0, scriptdirname) # add the diretcory to the path
#cwd = os.getcwd() # directoy from which the script has been lanched
#sys.path.insert(0, cwd) add the directory to the path
filedir = None                          # Directory used to save files
center = None                           # the solution center
sample = None                           # the solution samples
environment = None                      # the problem 
stepsize = 0.01                         # the learning stepsize
noiseStdDev = 0.02                      # the perturbation noise
sampleSize = 20                         # number of samples
wdecay = 0                              # wether we usse weight decay
sameenvcond = 0                         # whether population individuals experience the same conditions
maxsteps = 1000000                      # total max number of steps
evalCenter = 1                          # whether we evaluate the solution center
saveeach = 60                           # number of seconds after which we save data
saveeachg = 0                           # save pop data each n generations
fromgeneration = 0                      # start from generation   
nrobots = 1                             # number of robots
heterogeneous = 0                       # whether the parameters of robots are heterogeneous
algoname = "Salimans"                   # evolutionary algorithm

# Parse the [ADAPT] section of the configuration file
def parseConfigFile(filename):
    global maxsteps
    global envChangeEvery
    global environment
    global fullyRandom
    global stepsize
    global noiseStdDev
    global sampleSize
    global wdecay
    global sameenvcond
    global evalCenter
    global saveeach
    global nrobots
    global heterogeneous
    global algoname
    global saveeachg
    global fromgeneration

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        # Section EVAL
        options = config.options("ADAPT")
        for o in options:
            found = 0
            if o == "nrobots":
                nrobots = config.getint("ADAPT","nrobots")
                found = 1
            if o == "heterogeneous":
                heterogeneous = config.getint("ADAPT","heterogeneous")
                found = 1
            if o == "maxmsteps":
                maxsteps = config.getint("ADAPT","maxmsteps") * 1000000
                found = 1
            if o == "environment":
                environment = config.get("ADAPT","environment")
                found = 1
            if o == "stepsize":
                stepsize = config.getfloat("ADAPT","stepsize")
                found = 1
            if o == "noisestddev":
                noiseStdDev = config.getfloat("ADAPT","noiseStdDev")
                found = 1
            if o == "samplesize":
                sampleSize = config.getint("ADAPT","sampleSize")
                found = 1
            if o == "wdecay":
                wdecay = config.getint("ADAPT","wdecay")
                found = 1
            if o == "sameenvcond":
                sameenvcond = config.getint("ADAPT","sameenvcond")
                found = 1
            if o == "evalcenter":
                evalCenter = config.getint("ADAPT","evalcenter")
                found = 1
            if o == "saveeach":
                saveeach = config.getint("ADAPT","saveeach")
                found = 1
            if o == "saveeachg":
                saveeachg = config.getint("ADAPT","saveeachg")
                found = 1
            if o == "fromgeneration":
                saveeachg = config.getint("ADAPT","fromgeneration")
                found = 1
            if o == "algo":
                algoname = config.get("ADAPT","algo")
                found = 1
              
            if found == 0:
                print("\033[1mOption %s in section [ADAPT] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()
    else:
        print("\033[1mERROR: configuration file %s does not exist\033[0m" % (filename))
        sys.exit()

def start_train(argv):
    global maxsteps
    global environment
    global filedir
    global saveeach
    global nrobots
    global algoname

    print(argv)
    print(type(argv))
    print(type(argv[1]))

    argc = len(argv)

    # if called without parameters display help information
    if (argc == 1):
        helper()
        sys.exit(-1)

    # Default parameters:
    filename = None         # configuration file
    cseed = 1               # seed
    nreplications = 1       # nreplications
    filedir = './'          # directory
    testfile = None         # file containing the policy to be tested
    test = 0                # whether we rewant to test a policy (1=show behavior, 2=show neurons)
    displayneurons = 0      # whether we want to display the activation state of the neurons
    useTf = False           # whether we want to use tensorflow to implement the policy
    
    i = 1
    while (i < argc):
        if (argv[i] == "-f"):
            i += 1
            if (i < argc):
                filename = argv[i]
                i += 1
        elif (argv[i] == "-s"):
            i += 1
            if (i < argc):
                cseed = int(argv[i])
                i += 1
        elif (argv[i] == "-n"):
            i += 1
            if (i < argc):
                nreplications = int(argv[i])
                i += 1
        elif (argv[i] == "-a"):
            i += 1
            if (i < argc):
                algorithm = argv[i]
                i += 1
        elif (argv[i] == "-t"):
            i += 1
            test = 1
            if (i < argc):
                testfile = argv[i]
                i += 1
        elif (argv[i] == "-T"):
            i += 1
            test = 2
            displayneurons = 1
            if (i < argc):
                testfile = argv[i]
                i += 1   
        elif (argv[i] == "-d"):
            i += 1
            if (i < argc):
                filedir = argv[i]
                i += 1
        elif (argv[i] == "-tf"):
            i += 1
            useTf = True
        else:
            # We simply ignore the argument
            print("\033[1mWARNING: unrecognized argument %s \033[0m" % argv[i])
            i += 1

    # load the .ini file
    if filename is not None:
        parseConfigFile(filename)
    else:
        print("\033[1mERROR: You need to specify an .ini file\033[0m" % filename)
        sys.exit(-1)
    # if a directory is not specified, we use the current directory
    if filedir is None:
        filedir = scriptdirname

    # check whether the user specified a valid algorithm
    availableAlgos = ('CMAES','Salimans','xNES', 'sNES','SSS', 'pepg', 'coevo2', 'coevo2r')
    if algoname not in availableAlgos:
        print("\033[1mAlgorithm %s is unknown\033[0m" % algoname)
        print("Please use one of the following algorithms:")
        for a in availableAlgos:
            print("%s" % a)
        sys.exit(-1)

    print("Environment %s nreplications %d maxmsteps %dm " % (environment, nreplications, maxsteps / 1000000))
    env = None
    policy = None
    


    # Evorobot Environments (we expect observation and action made of numpy array of float32)
    if "Er" in environment:
        ErProblem = __import__(environment)
        env = ErProblem.PyErProblem()
        # Create a new doublepole object
        #action_space = spaces.Box(-1., 1., shape=(env.noutputs,), dtype='float32')
        #observation_space = spaces.Box(-np.inf, np.inf, shape=(env.ninputs,), dtype='float32')
        ob = np.arange(env.ninputs * nrobots, dtype=np.float32)
        ac = np.arange(env.noutputs * nrobots, dtype=np.float32)
        done = np.arange(1, dtype=np.int32)
        env.copyObs(ob)
        env.copyAct(ac)
        env.copyDone(done)      
        from policy import ErPolicy
        policy = ErPolicy(env, env.ninputs, env.noutputs, env.low, env.high, ob, ac, done, filename, cseed, nrobots, heterogeneous, test)

    # Bullet environment (we expect observation and action made of numpy array of float32)
    if "Bullet" in environment:
        import gym
        from gym import spaces
        import pybullet
        import pybullet_envs
        env = gym.make(environment)
        # Define the objects required (they depend on the environment)
        ob = np.arange(env.observation_space.shape[0], dtype=np.float32)
        ac = np.arange(env.action_space.shape[0], dtype=np.float32)
        # Define the policy
        from policy import BulletPolicy
        policy = BulletPolicy(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], ob, ac, filename, cseed, nrobots, heterogeneous, test)

   # Gym environment (we expect observation and action made of numpy array of float64 or discrete actions)
    if (not "Bullet" in environment) and (not "Er" in environment):
        import gym
        from gym import spaces
        env = gym.make(environment)
        # Define the objects required (they depend on the environment)
        ob = np.arange(env.observation_space.shape[0], dtype=np.float32)
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            ac = np.arange(env.action_space.shape[0], dtype=np.float32)
        else:
            ac = np.arange(env.action_space.n, dtype=np.float32)        
        # Define the policy
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            from policy import GymPolicy
            policy = GymPolicy(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], ob, ac, filename, cseed, nrobots, heterogeneous, test)
        else:
            from policy import GymPolicyDiscr
            policy = GymPolicyDiscr(env, env.observation_space.shape[0], env.action_space.n, 0.0, 0.0, ob, ac, filename, cseed, nrobots, heterogeneous, test)           

    policy.environment = environment
    policy.saveeach = saveeach
    

    # Create the algorithm class
    if (algoname == 'CMAES'):
        from cmaes import CMAES
        algo = CMAES(env, policy, cseed, filedir)
    elif (algoname =='Salimans'):
        from salimans import Salimans
        algo = Salimans(env, policy, cseed, filedir)
    elif (algoname == 'xNES'):
        from xnes import xNES
        algo = xNES(env, policy, cseed, filedir)
    elif (algoname == 'sNES'):
        from snes import sNES
        algo = sNES(env, policy, cseed, filedir)
    elif (algoname == 'SSS'):
        from sss import SSS
        algo = SSS(env, policy, cseed, filedir)
    elif (algoname == 'coevo2'):
        from coevo2 import coevo2
        algo = coevo2(env, policy, cseed, filedir)
    elif (algoname == 'coevo2r'):
        from coevo2r import coevo2
        algo = coevo2(env, policy, cseed, filedir)
    elif (algoname == 'pepg'):
        from pepg import pepg
        algo = pepg(env, policy, cseed, filedir)
    # Set evolutionary variables
    algo.setEvoVars(sampleSize, stepsize, noiseStdDev, sameenvcond, wdecay, evalCenter, saveeachg, fromgeneration)

    if (test > 0):
        # test a policy
        print("Run Test: Environment %s testfile %s" % (environment, testfile))
        algo.test(testfile)
    else:
        # run evolution
        if (cseed != 0):
            print("Run Evolve: Environment %s Seed %d Nreplications %d" % (environment, cseed, nreplications))
            for r in range(nreplications):
                algo.run(maxsteps)
                algo.seed += 1
                policy.seed += 1
                algo.reset()
                policy.reset()
        else:
            print("\033[1mPlease indicate the seed to run evolution\033[0m")

def generate_seed(set_state,states):
    seed=0
    for i in range(len(set_state)):
        state=set_state[i]
        for j in range(len(states)):
            if abs(state-states[j][i])<0.0000001:
                seed+=(j+1)*10**i
                break
    return seed

if __name__ == "__main__":
    iterations=300
    n=5**6
    seeds=[i for i in range(1,n+1)]
    random.shuffle(seeds)

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

    env_states=pd.read_csv("/opt/evorobotpy/lib/environment_states.csv",header=None)
    env_states=env_states.values.tolist()

    for i in range(iterations):
        env_states_old=pd.read_csv("/opt/evorobotpy/xdpole/environment_states_ranges.csv",header=None)
        env_states_old=env_states_old.values.tolist()
        
        seeds=[j for j in range(1,n+1)]
        for j in range(len(env_states_old)):
            seeds.remove(env_states_old[j][0])
        
        set_seed=random.choice(seeds)

        env_states_old.append(env_states[set_seed])
        set_state=env_states[set_seed][1:]
        seed=generate_seed(set_state,states)
        
        print(i,len(env_states_old),seed)
        df = pd.DataFrame.from_records(env_states_old)
        df.to_csv("/opt/evorobotpy/xdpole/environment_states_ranges.csv",header=False,index=False)

        m=['/opt/evorobotpy/bin/Project_train_NbyN.py','-f','/opt/evorobotpy/xdpole/ErDpole.ini', '-s', str(seed)]
        start_train(m)

