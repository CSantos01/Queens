from __init__ import GameBoard, Queen
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from pathlib import Path


# Board Settings
BOARD_SIZE = 7
NUM_COLORS = 5
MAX_EPISODES = 1000


class QueenEnv:
    def __init__(self, size=BOARD_SIZE, num_colors=NUM_COLORS):
        self.size = size
        self.num_colors = num_colors
        self.board = GameBoard(size=size, num_colors=num_colors)
        self.reset()

    def reset(self):
        """Resets the board"""
        self.board.reset_queens()
        self.queens = []
        return self.get_state()

    def get_state(self):
        """Returns a 3-channel representation of the board"""
        attack_zones = self.get_attack_zones()
        color_list = self.board.colors
        color_map = {color: i for i, color in enumerate(color_list)}
        colors = np.vectorize(color_map.get)(self.board.get_colors())
        queens = np.zeros((self.size, self.size))
        for queen in self.queens:
            queens[queen.y][queen.x] = 1
        return np.stack([queens, colors, attack_zones], axis=0)

    def get_attack_zones(self):
        """Computes attacked positions"""
        attack_zones = np.zeros((self.size, self.size))
        for queen in self.queens:
            qx, qy = queen.x, queen.y
            for i in range(self.size):
                attack_zones[qy][i] = 1  # Row
                attack_zones[i][qx] = 1  # Column
            if 0 <= qy + 1 < self.size and 0 <= qx + 1 < self.size:
                attack_zones[qy + 1][qx + 1] = 1  # Diagonal
            if 0 <= qy - 1 < self.size and 0 <= qx - 1 < self.size:
                attack_zones[qy - 1][qx - 1] = 1  # Diagonal
            if 0 <= qy + 1 < self.size and 0 <= qx - 1 < self.size:
                attack_zones[qy + 1][qx - 1] = 1  # Anti-diagonal
            if 0 <= qy - 1 < self.size and 0 <= qx + 1 < self.size:
                attack_zones[qy - 1][qx + 1] = 1  # Anti-diagonal
        return attack_zones

    def step(self, action):
        """Takes an action (place queen) and returns new state, reward, and done flag"""
        x, y = divmod(action, self.size)
        if self.board.board[y][x].queen or self.get_attack_zones()[y][x] == 1:
            return self.get_state(), -1, True  # Invalid move

        queen = Queen(self.board, x, y)
        self.queens.append(queen)
        done = len(self.queens) == self.size  # Game is over when all queens are placed
        reward = 10 if done else 1
        return self.get_state(), reward, done

    def valid_moves(self):
        return [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if not self.board.board[y][x].queen and self.get_attack_zones()[y][x] == 0
        ]


# Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def training(board_size=BOARD_SIZE, num_colors=NUM_COLORS, max_episodes=MAX_EPISODES):
    # Training Loop
    env = QueenEnv()
    dqn = DQN(BOARD_SIZE * BOARD_SIZE * 3, BOARD_SIZE * BOARD_SIZE)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    memory = deque(maxlen=10000)
    GAMMA = 0.9
    EPSILON = 1.0
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01

    for episode in range(MAX_EPISODES):
        state = env.reset()
        state = torch.FloatTensor(state.astype(np.float32).flatten()).unsqueeze(0)
        total_reward = 0

        for step in range(BOARD_SIZE**2):
            if random.random() < EPSILON:
                action = random.randint(0, BOARD_SIZE * BOARD_SIZE - 1)
            else:
                with torch.no_grad():
                    action = torch.argmax(dqn(state)).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
            memory.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            if done:
                break

        if len(memory) > 32:
            batch = random.sample(memory, 32)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.cat(states)
            next_states = torch.cat(next_states)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = dqn(states)
            next_q_values = dqn(next_states).detach()
            target_q_values = rewards + GAMMA * next_q_values.max(dim=1)[0] * (
                1 - dones
            )
            loss = F.mse_loss(
                q_values.gather(1, actions.unsqueeze(1)).squeeze(), target_q_values
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

    print("Training completed!")
    return dqn


if __name__ == "__main__":
    dqn = training()
    this_dir = Path(__file__).parent
    torch.save(dqn.state_dict(), this_dir / "dqn_model.pth")
