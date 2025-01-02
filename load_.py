import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import time
from wrappers import wrapper_
from stable_baselines3 import PPO

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
env = wrapper_(env)

model = PPO.load('utils/n1024b64l3_10000000_steps.zip')

obs = env.reset()
total_reward = 0
finished = False
while finished != True:
    env.render()
    time.sleep(0.1)
    action, state = model.predict(obs)
    obs, reward, finished, info = env.step(action.item())
    total_reward += reward
print('Total Reward:', total_reward)
env.close()
