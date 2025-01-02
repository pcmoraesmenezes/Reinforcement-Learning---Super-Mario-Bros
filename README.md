
# Super Mario Bros AI with Reinforcement Learning

## Project Overview

This project leverages **Deep Reinforcement Learning (DRL)** to train an AI agent capable of playing the classic game **Super Mario Bros**. The implementation utilizes **Proximal Policy Optimization (PPO)**, a popular DRL algorithm, to optimize the agent's behavior. 

The project integrates custom environment wrappers, preprocessing steps, and modular code design to improve performance, learning efficiency, and model scalability. It was developed to enhance knowledge in artificial intelligence, specifically in reinforcement learning applied to complex video game environments.

## Features

- **Custom Environment Wrappers:** Includes wrappers for reward clipping, frame stacking, action skipping, and grayscale image conversion to enhance learning efficiency.
- **Modular Codebase:** Designed with clear modularity to separate training, environment setup, and model loading.
- **Pre-trained Model:** A trained PPO model (`n1024b64l3_10000000_steps.zip`) included in the project to demonstrate the agent's performance.
- **High Scalability:** Uses parallelized environments for efficient training with **SubprocVecEnv**.
- **Seamless Integration with Gym and NES-Py Libraries:** Supports Gym's Super Mario Bros environment with custom JoypadSpace actions.
- **Visualization:** The agent's gameplay can be rendered during inference for real-time observation.

## File Structure

```plaintext
.
├── wrappers.py              # Contains custom environment wrappers
├── load_.py                 # Script to load and test the trained model
├── enviroment_.py           # Main training script for the PPO model
├── utils/
│   ├── requirements.txt     # Python dependencies
│   ├── n1024b64l3_10000000_steps.zip  # Pre-trained PPO model weights
└── README.md                # Project documentation
```

## Environment Setup

The environment setup is a crucial aspect of this project. Wrappers were implemented to preprocess the game environment to facilitate faster learning:

1. **Frame Skipping and Max Pooling:** Reduces the number of frames the agent processes by skipping frames and taking the maximum pixel value between frames.
2. **Frame Resizing and Grayscale Conversion:** Converts frames to 84x84 grayscale images, reducing the input size and computation requirements.
3. **Reward Clipping:** Normalizes rewards to the range {+1, 0, -1}.
4. **Frame Stacking:** Stacks the last four frames together to capture temporal information for the agent.

## Training Configuration

The model training process uses the following configurations:

- **Algorithm:** PPO
- **Policy:** CnnPolicy
- **Parallel Environments:** 8
- **Hyperparameters:**
  - `gamma`: 0.99
  - `n_steps`: 1024
  - `batch_size`: 64
  - `learning_rate`: 0.0003
  - `vf_coef`: 0.5
  - `ent_coef`: 0.01
- **Total Timesteps:** 10,000,000

Training runs are checkpointed every 6250 timesteps, and the training time for the model is logged.

## Requirements

To run this project, install the required Python dependencies:

```bash
pip install -r utils/requirements.txt
```

**Key Libraries:**
- `gym`
- `gym_super_mario_bros`
- `stable-baselines3`
- `opencv-python`

## Usage

### Training the Model

To train the PPO model, run the `enviroment_.py` script:

```bash
python enviroment_.py
```

This script initializes the environment, applies the custom wrappers, and trains the agent using PPO. Training logs and checkpoints will be saved in the `logs/` directory.

### Testing the Trained Model

To evaluate the pre-trained model, use the `load_.py` script:

```bash
python load_.py
```

This script loads the saved model (`n1024b64l3_10000000_steps.zip`) and runs the agent in the environment to showcase its performance.

## Results

The agent achieves high scores in **Super Mario Bros - Level 1-1** using the **RIGHT_ONLY** action space. This restricted action space simplifies the problem, allowing the agent to focus on moving forward and jumping to complete the level.

### Total Reward:
The total reward is displayed at the end of each evaluation session.

## Future Improvements

- Expanding to more complex action spaces (e.g., `SIMPLE_MOVEMENT` or `COMPLEX_MOVEMENT`).
- Enhancing the reward structure to incentivize better exploration.
- Experimenting with advanced RL algorithms, such as **A2C** or **DQN**, to compare performance.
- Adding support for multi-level training and transfer learning across levels.

## Conclusion

This project is a hands-on application of **Deep Reinforcement Learning** to a classic game environment. It demonstrates the potential of AI to learn and solve complex sequential tasks through experience and rewards. The modular design and implementation serve as a foundation for further exploration in AI and reinforcement learning.
