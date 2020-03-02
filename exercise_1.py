import gym
from time import sleep

t=5
env = gym.make('CartPole-v0')
env.reset()
for _ in range(200):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    sleep(t/200)

env.close()

env = gym.make('Pendulum-v0')
env.reset()
for _ in range(200):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    sleep(t/200)

env.close()