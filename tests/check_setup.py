import sys
import torch
import numpy as np
import matplotlib
import gymnasium as gym

def main():
    # 1. Print Python version
    print(f"Python version: {sys.version}")

    # 2. Print imported library versions
    print(f"torch version: {torch.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"matplotlib version: {matplotlib.__version__}")
    print(f"gymnasium version: {gym.__version__}")

    # 3. Print torch.__version__ and torch.cuda.is_available()
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    # 4. Random tensor (3, 3), multiplied by itself, print result shape
    tensor = torch.rand(3, 3)
    product = torch.matmul(tensor, tensor)
    print(f"Tensor product shape: {product.shape}")

    # 5. Gymnasium CartPole-v1 environment, reset(seed=0), 5 random steps with seeded action space
    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    env.action_space.seed(0)
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i + 1} observation shape: {obs.shape}")
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

    # 6. Success message
    print("SETUP CHECK PASSED")

if __name__ == "__main__":
    main()
