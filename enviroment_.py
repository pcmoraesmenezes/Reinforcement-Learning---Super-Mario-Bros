import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
import time
from wrappers import wrapper_
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def create_env(env_name):
    def _init():
        env = gym_super_mario_bros.make(env_name)
        env = JoypadSpace(env, RIGHT_ONLY)
        env = wrapper_(env)

        original_reset = env.reset

        def custom_reset(*args, **kwargs):
            kwargs.pop('options', None)
            kwargs.pop('seed', None)
            result = original_reset(*args, **kwargs)
            if not isinstance(result, tuple):
                return result, {}
            return result

        original_step = env.step

        def custom_step(action):
            result = original_step(action)
            if len(result) == 4:
                obs, reward, done, info = result
                terminated = done
                truncated = info.get('TimeLimit.truncated', False)
                return obs, reward, terminated, truncated, info
            return result

        env.reset = custom_reset
        env.step = custom_step
        env.render_mode = None
        return env
    return _init


if __name__ == '__main__':
    cpu_threads = 8
    env = SubprocVecEnv([create_env(env_name='SuperMarioBros-1-1-v0') for i in range(cpu_threads)])

    checkpoint_callback = CheckpointCallback(
        save_freq=6250,
        save_path='logs/',
        name_prefix='n1024b64l3'
    )

    model = PPO(
        policy='CnnPolicy',
        env=env,
        gamma=0.99,
        n_steps=1024,
        batch_size=64,
        learning_rate=0.0003,
        vf_coef=0.5,
        ent_coef=0.01,
        verbose=1,
        device='cpu'
    )

    start_time = time.time()
    model.learn(total_timesteps=10000000, log_interval=10, callback=checkpoint_callback)
    end_time = time.time()

    print('Total training time in hours:', (end_time - start_time) / 3600)
