import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# Base file path for model saves and related files. Change this if you want
# models saved to a different location. Defaults to the current source folder.
FILEPATH = os.path.dirname(os.path.abspath(__file__))

class SimpleDQN(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=8):
        super(SimpleDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleDQNAgent:
    def __init__(self, state_size=15, action_size=8):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SimpleDQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.load_model()
    
    def get_state_representation(self, game_state, my_agent, opponent):
        """Better state representation with more features"""
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        state = []
        
        # 1. Immediate danger in 4 directions (4 values)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            is_safe = float(game_state.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos)
            state.append(is_safe)
        
        # 2. Extended danger (2 steps ahead in each direction) (4 values)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pos1 = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            pos2 = ((pos1[0] + dx) % 20, (pos1[1] + dy) % 18)
            danger_level = 0
            if game_state.board.get_cell_state(pos1) != 0 and pos1 != opp_pos:
                danger_level += 0.5
            if game_state.board.get_cell_state(pos2) != 0 and pos2 != opp_pos:
                danger_level += 0.5
            state.append(danger_level)
        
        # 3. Relative position and distance (3 values)
        dx = (opp_pos[0] - my_pos[0]) % 20
        if dx > 10: dx = dx - 20
        dy = (opp_pos[1] - my_pos[1]) % 18
        if dy > 9: dy = dy - 18
        distance = abs(dx) + abs(dy)
        
        state.extend([dx / 10.0, dy / 9.0, distance / 20.0])
        
        # 4. Boost information (2 values)
        state.append(float(my_agent.boosts_remaining > 0))
        state.append(my_agent.boosts_remaining / 3.0)
        
        # 5. Safe moves and strategic info (2 values)
        safe_moves = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
                safe_moves += 1
        state.append(safe_moves / 4.0)
        
        # Check if opponent is trapped
        opp_safe_moves = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((opp_pos[0] + dx) % 20, (opp_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == my_pos:
                opp_safe_moves += 1
        state.append(opp_safe_moves / 4.0)
        
        return np.array(state, dtype=np.float32)
    
    def get_valid_actions(self, game_state, my_agent, boosts_remaining, opponent):
        """Get valid action indices - prevents reverse moves and checks safety.

        Args:
            game_state: the Game instance (board access)
            my_agent: the agent whose moves we're computing
            boosts_remaining: integer boosts left for my_agent
            opponent: the opposing agent

        Returns:
            List of valid action indices (0-7)
        """
        valid_actions = []

        from case_closed_game import Direction

        current_direction = my_agent.direction
        reverse_directions = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        forbidden_direction = reverse_directions.get(current_direction)

        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        my_pos = my_agent.trail[-1]

        for dir_idx, direction in enumerate(directions):
            # Never allow reverse moves
            if direction == forbidden_direction:
                continue

            dx, dy = direction.value
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)

            # Check if move is safe (empty cell or moves into opponent head)
            is_safe = (game_state.board.get_cell_state(next_pos) == 0) or (next_pos == opponent.trail[-1])

            if is_safe:
                # Action without boost
                valid_actions.append(dir_idx * 2 + 0)

                # Action with boost (only if both steps are safe)
                if boosts_remaining > 0:
                    boost_pos = ((next_pos[0] + dx) % 20, (next_pos[1] + dy) % 18)
                    boost_safe = (game_state.board.get_cell_state(boost_pos) == 0) or (boost_pos == opponent.trail[-1])

                    if boost_safe:
                        valid_actions.append(dir_idx * 2 + 1)

        # If no valid actions, allow at least the non-reverse directions (even if unsafe)
        if not valid_actions:
            for dir_idx, direction in enumerate(directions):
                if direction != forbidden_direction:
                    valid_actions.append(dir_idx * 2 + 0)  # Only non-boost as fallback

        return valid_actions
        
    def act(self, state, valid_actions):
        """Choose action with better safety checks"""
        # If there are no valid actions, fall back to a safe default (no-boost action 0)
        if not valid_actions:
            # No valid actions available; return a deterministic fallback
            return 0
        
        if np.random.random() <= self.epsilon:
            # During exploration: heavily favor non-boost actions
            non_boost_actions = [a for a in valid_actions if a % 2 == 0]
            if non_boost_actions and (np.random.random() < 0.9 or not valid_actions):
                return random.choice(non_boost_actions)
            return random.choice(valid_actions)
        else:
            # Exploitation
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                q_values = q_values.cpu().numpy()[0]
            
            # Only consider valid actions
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Convert memory to tensors efficiently
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # In simple_dqn_agent.py, improve the reward function:
    def get_reward(self, my_agent, opponent, prev_my_alive, prev_opp_alive, game_state, action_used):
        """Comprehensive reward function with diversity bonus"""
        reward = 0
        
        # MASSIVE Game outcome rewards - clear signal what matters most
        if not my_agent.alive and prev_my_alive:
            return -100.0  # Death is terrible
        
        if not opponent.alive and prev_opp_alive:
            return 100.0   # Winning is awesome
        
        # Base survival reward (small positive for staying alive)
        reward += 0.1
        
        # Positioning and distance rewards
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        dx = (opp_pos[0] - my_pos[0]) % 20
        if dx > 10: dx = dx - 20
        dy = (opp_pos[1] - my_pos[1]) % 18
        if dy > 9: dy = dy - 18
        distance = abs(dx) + abs(dy)
        
        # Reward for optimal attacking distance (not too close, not too far)
        if 2 <= distance <= 6:
            reward += 0.5  # Good positioning for attack
        elif distance < 2:
            reward -= 0.2  # Too close can be dangerous
        
        # Safety assessment - CRITICAL for survival
        safe_moves = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
                safe_moves += 1
        
        # Safety rewards/penalties
        if safe_moves >= 3:
            reward += 0.3  # Good - plenty of options
        elif safe_moves == 2:
            reward += 0.1  # Okay
        elif safe_moves == 1:
            reward -= 0.5  # Dangerous - almost trapped
        else:
            reward -= 2.0  # Critical - completely trapped
        
        # DIVERSITY BONUS (instead of repetition penalty)
        if len(my_agent.trail) >= 4:
            recent_directions = []
            for i in range(1, min(4, len(my_agent.trail))):
                curr = my_agent.trail[-i]
                prev = my_agent.trail[-(i+1)]
                dx = (curr[0] - prev[0]) % 20
                dy = (curr[1] - prev[1]) % 18
                
                # Normalize for torus wrapping
                if dx == 19: dx = -1
                if dx == -19: dx = 1
                if dy == 17: dy = -1  
                if dy == -17: dy = 1
                
                if dx == 1: recent_directions.append("RIGHT")
                elif dx == -1: recent_directions.append("LEFT")
                elif dy == 1: recent_directions.append("DOWN") 
                elif dy == -1: recent_directions.append("UP")
            
            # Reward movement diversity
            if len(recent_directions) >= 3:
                unique_directions = len(set(recent_directions))
                if unique_directions == 3:
                    reward += 1.0  # Bonus for varied movement
                elif unique_directions >= 2:
                    reward += 0.3  # Small bonus for some variety
        
        # Strategic boost usage
        direction, used_boost = action_used
        if used_boost:
            # Only reward boost usage in good situations
            if distance < 5 and safe_moves >= 2:
                reward += 0.8  # Good aggressive boost
            elif safe_moves <= 1:
                reward -= 1.0  # Bad - boosting when trapped
            else:
                reward -= 0.3  # Wasteful boost
        
        # Small reward for having boosts available (encourage strategic use)
        if my_agent.boosts_remaining == 3:
            reward += 0.2
        elif my_agent.boosts_remaining == 2:
            reward += 0.1
        
        # Penalty for obvious oscillation (going back and forth)
        if len(my_agent.trail) >= 4:
            positions = [my_agent.trail[-1], my_agent.trail[-2], my_agent.trail[-3], my_agent.trail[-4]]
            if len(set(positions)) <= 2:  # Only 2 unique positions in last 4 moves
                reward -= 1.5  # Penalize obvious oscillation
        
        return reward
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\simple_dqn_model.pth")
        print(f"Model saved (epsilon: {self.epsilon:.3f})")
    
    def load_model(self):
        try:
            checkpoint = torch.load(r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\simple_dqn_model.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded (epsilon: {self.epsilon:.3f})")
        except FileNotFoundError:
            print("No model found, starting fresh")