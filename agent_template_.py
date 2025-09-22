import gymnasium as gym  # Gym for the Lunar Lander environment
import numpy as np       # Numerical operations
import torch             # PyTorch for neural networks
import torch.nn as nn    # Neural network modules
import torch.optim as optim  # Optimization algorithms
from collections import deque  # Efficient data structure for experience replay
import matplotlib.pyplot as plt # Plotting
import random            # Random number generation

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
class DQN(nn.Module):
    """
    Deep Q-Network: Neural network model that approximates the Q-value function.
    """
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # Define the network architecture
        self.fc1 = nn.Linear(state_size, 64)   # First fully connected layer
        self.fc2 = nn.Linear(64, 64)         # Second fully connected layer
        self.fc3 = nn.Linear(64, action_size)  # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))    # Apply ReLU activation to first layer
        x = torch.relu(self.fc2(x))    # Apply ReLU activation to second layer
        x = self.fc3(x)                # Output Q-values for each action
        return x

class LunarLanderAgent:
    def __init__(self):
        """
        Initialize the agent.
        """
        # Initialize the Lunar Lander environment
        self.env = gym.make('LunarLander-v3')

        # Define the size of the state and action spaces
        self.state_size = 8   # State has 8 dimensions
        self.action_size = 4  # Four discrete actions

        # Hyperparameters for learning
        self.learning_rate = 0.0005  # Learning rate for the optimizer
        self.gamma = 0.99          # Discount factor for future rewards

        # Epsilon parameters for epsilon-greedy policy
        self.epsilon = 1.0         # Initial exploration rate
        self.epsilon_min = 0.01    # Minimum exploration rate
        self.epsilon_decay = 0.995 # Decay rate for epsilon

        # Experience replay parameters
        self.batch_size = 128       
        self.memory_size = 100000  # Maximum size of the replay buffer
        self.memory = deque(maxlen=self.memory_size)  # Replay buffer

        # Set up the device for computation (select CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the policy network (DQN)
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)

        # Initialize the target network
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        self.target_net.eval()  # Set target network to evaluation mode

        # Set up the optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber Loss

        # Parameters for updating the target network
        self.update_target_every = 5  # Update target network every N episodes

        # Variables for tracking progress
        self.best_average_reward = -float('inf')  # Best average reward achieved

        # Set training mode to False for submission (inference mode)
        self.training = False

    def select_action(self, state):
        """
        Select an action based on the current state.
        """
        if self.training:
            epsilon = self.epsilon  # Use current epsilon during training
        else:
            epsilon = 0.0  # No exploration during testing/inference

        if np.random.rand() <= epsilon:
            # Exploration: select a random action
            action = random.randrange(self.action_size)
        else:
            # Exploitation: select the action with the highest Q-value
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert state to tensor
            with torch.no_grad():
                q_values = self.policy_net(state)  # Get Q-values from the policy network
            action = torch.argmax(q_values).item()  # Choose the action with the highest Q-value

        return action

    def train(self, num_episodes):
        """
        Train the agent over a specified number of episodes.
        """
        # Enable training mode
        self.training = True

        scores_window = deque(maxlen=100)  # Keep track of rewards over the last 100 episodes
        self.episode_rewards = []          # Total rewards per episode
        self.average_rewards = []          # Average rewards over the last 100 episodes

        for i_episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()  # Reset the environment for a new episode
            total_reward = 0
            done = False

            while not done:
                # Select an action using the epsilon-greedy policy
                action = self.select_action(state)

                # Perform the action in the environment
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated  # Check if the episode is done

                # Store the transition in the replay buffer
                self.memory.append((state, action, reward, next_state, done))

                # Update the agent's networks
                self.update()

                # Move to the next state
                state = next_state
                total_reward += reward  # Accumulate the reward

            # Decay epsilon to reduce exploration over time
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

            # Save the total reward for this episode
            self.episode_rewards.append(total_reward)
            scores_window.append(total_reward)  # Update the scores window
            average_reward = np.mean(scores_window)
            self.average_rewards.append(average_reward)

            # Print progress every 10 episodes
            if i_episode % 10 == 0:
                print(f"Episode {i_episode}/{num_episodes}, "
                      f"Average Reward: {average_reward:.2f}, Epsilon: {self.epsilon:.2f}")

            # Save the model if performance improves and exploration is low
            if average_reward > self.best_average_reward and self.epsilon < 0.1:
                print(f"New best average reward: {average_reward:.2f} at episode {i_episode}, saving model...")
                self.best_average_reward = average_reward
                self.save_model('best_model_dqn.pth')

            # Update the target network periodically
            if i_episode % self.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # Plot the learning curve after training
        plt.plot(self.average_rewards)
        plt.title('DQN: Average Reward over Training Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward (Last 100 Episodes)')
        plt.show()

    def update(self):
        """
        Update the agent's policy network using a batch of experiences.
        """
        # Wait until there is enough experience in the replay buffer
        if len(self.memory) < self.batch_size:
            return

        # Sample a random minibatch of transitions
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract states, actions, rewards, next_states, and dones from the minibatch
        states = torch.FloatTensor(np.array([transition[0] for transition in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([transition[1] for transition in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([transition[2] for transition in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([float(transition[4]) for transition in minibatch])).unsqueeze(1).to(self.device)

        # Compute the current Q-values using the policy network
        q_values = self.policy_net(states).gather(1, actions)

        # Compute the target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss between current Q-values and target Q-values
        loss = self.criterion(q_values, target_q_values)

        # Perform a gradient descent step to minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def test(self, num_episodes=100):
        """
        Evaluate the agent's performance over a specified number of episodes.
        """
        self.training = False
        self.epsilon = 0.0     # Disable exploration
        total_rewards = []

        for i_episode in range(num_episodes):
            state, _ = self.env.reset()  # Reset the environment
            total_reward = 0
            done = False

            while not done:
                # Select action based on the learned policy
                action = self.select_action(state)

                # Perform the action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Move to the next state
                state = next_state
                total_reward += reward  # Accumulate the reward

            total_rewards.append(total_reward)
            print(f"Episode {i_episode + 1}/{num_episodes}, Reward: {total_reward:.2f}")

        # Calculate and print the average reward over all test episodes
        average_reward = np.mean(total_rewards)
        print(f"Average reward over {num_episodes} test episodes: {average_reward:.2f}")

    def save_model(self, file_name):
        """
        Save the agent's policy network to a file.
        """
        torch.save(self.policy_net.state_dict(), file_name)
        print(f"Model saved to {file_name}.")

    def load_model(self, file_name):
        """
        Load the agent's policy network from a file.
        """
        # Load the saved model weights
        self.policy_net.load_state_dict(torch.load(file_name, map_location=self.device))
        self.policy_net.to(self.device)

        # Update the target network to match the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Model loaded from {file_name}.")
        
#uncomment to train:
"""
if __name__ == '__main__':
     agent = LunarLanderAgent()
     agent_model_file = 'best_model_dqn.pth'  # File name for saving/loading the model

     # Train the agent
     num_training_episodes = 2000  # Number of episodes for training
     print("Training the DQN agent...")
     agent.train(num_training_episodes)
     print("Training completed.")

     # Save the trained model
     agent.save_model(agent_model_file)
     print("Model saved.")

     # Load the trained model
     agent.load_model(agent_model_file)

     # Test the agent
     print("Testing the DQN agent...")
     agent.test(num_episodes=100)
"""

