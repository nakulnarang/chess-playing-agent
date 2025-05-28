import gym
import gym_chess
import random

# Create the chess environment
env = gym.make('Chess-v0')

# Reset the environment to the initial state
obs = env.reset()
print("Initial observation (board state):")
print(obs)

# Get list of legal moves from the environment
legal_moves = env.legal_moves
print("\nLegal moves:", legal_moves)

# Take a random legal move
if legal_moves:
    action = random.choice(legal_moves)
    print("\nSelected random move:", action)

    # Perform the action and get the new state, reward, done flag, and info
    new_obs, reward, done, info = env.step(action)
    print("\nNew board state after move:")
    print(new_obs)
    print("Reward:", reward)
    print("Game finished:", done)
    print("Additional info:", info)
else:
    print("No legal moves available!")

# Close the environment
env.close()
