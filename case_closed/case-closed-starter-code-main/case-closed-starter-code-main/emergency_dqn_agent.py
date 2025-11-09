import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from heapq import heappush, heappop

# Emergency Pathfinding DQN Agent - COMPATIBLE with current SimpleDQNAgent
class EmergencyDQN(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=8):
        super(EmergencyDQN, self).__init__()
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

class EmergencyPathfindingAgent:
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
        self.model = EmergencyDQN(state_size, 256, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Emergency pathfinding parameters
        self.danger_threshold = 1  # Switch to pathfinding when safe moves <= this
        self.last_emergency_action = None
        
        self.load_model()
    
    def is_in_danger(self, game_state, my_agent, opponent):
        """Check if agent is in immediate danger and needs emergency pathfinding"""
        my_pos = my_agent.trail[-1]
        
        # Count safe moves
        safe_moves = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == opponent.trail[-1]:
                safe_moves += 1
        
        # Emergency if we have very few safe moves
        if safe_moves <= self.danger_threshold:
            return True, safe_moves
        
        # Emergency if we're about to be trapped (check 2 steps ahead)
        future_danger = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            pos1 = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            pos2 = ((pos1[0] + dx) % 20, (pos1[1] + dy) % 18)
            
            # If both steps are dangerous, this direction is risky
            danger1 = game_state.board.get_cell_state(pos1) != 0 and pos1 != opponent.trail[-1]
            danger2 = game_state.board.get_cell_state(pos2) != 0 and pos2 != opponent.trail[-1]
            
            if danger1 and danger2:
                future_danger += 1
        
        # If most directions lead to danger soon, trigger emergency
        if future_danger >= 2:
            return True, safe_moves
        
        return False, safe_moves
    
    def find_emergency_escape(self, game_state, my_agent, opponent):
        """Find the safest escape route when in danger"""
        my_pos = my_agent.trail[-1]
        
        # Try to find the safest area on the board
        safest_position = self.find_safest_area(game_state, my_pos, opponent)
        
        # Use A* to find path to safest area
        escape_path = self.emergency_a_star(my_pos, safest_position, game_state)
        
        if escape_path and len(escape_path) > 0:
            next_pos = escape_path[0]
            dx = (next_pos[0] - my_pos[0]) % 20
            dy = (next_pos[1] - my_pos[1]) % 18
            
            # Normalize for torus wrapping
            if dx == 19: dx = -1
            if dx == -19: dx = 1
            if dy == 17: dy = -1
            if dy == -17: dy = 1
            
            # Map to direction and action
            if dx == 1: return 0  # RIGHT
            if dx == -1: return 1  # LEFT
            if dy == 1: return 2  # DOWN
            if dy == -1: return 3  # UP
        
        # Fallback: find any safe move
        return self.find_any_safe_move(game_state, my_agent, opponent)
    
    def find_safest_area(self, game_state, my_pos, opponent):
        """Find the position with the most empty space around it"""
        best_pos = my_pos
        best_score = -9999
        
        # Sample positions around the board
        sample_positions = []
        for x in range(20):
            for y in range(18):
                if game_state.board.get_cell_state((x, y)) == 0:
                    sample_positions.append((x, y))
        
        # Limit sampling for performance
        if len(sample_positions) > 50:
            sample_positions = random.sample(sample_positions, 50)
        
        for pos in sample_positions:
            # Score based on empty space and distance from opponent
            empty_space = self.count_empty_space(game_state, pos, radius=2)
            opp_distance = self.torus_manhattan(pos, opponent.trail[-1])
            
            score = empty_space + (opp_distance * 0.5)
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
        return best_pos
    
    def count_empty_space(self, game_state, center, radius=2):
        """Count empty cells around a position"""
        empty_count = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                check_pos = (
                    (center[0] + dx) % 20,
                    (center[1] + dy) % 18
                )
                
                if game_state.board.get_cell_state(check_pos) == 0:
                    empty_count += 1
        
        return empty_count
    
    def torus_manhattan(self, pos1, pos2):
        """Calculate Manhattan distance on torus board"""
        dx = min(abs(pos1[0] - pos2[0]), 20 - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), 18 - abs(pos1[1] - pos2[1]))
        return dx + dy
    
    def emergency_a_star(self, start, goal, game_state):
        """A* pathfinding optimized for emergency escape"""
        def heuristic(a, b):
            return self.torus_manhattan(a, b)
        
        def get_safe_neighbors(pos):
            neighbors = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = ((pos[0] + dx) % 20, (pos[1] + dy) % 18)
                # In emergency, only consider truly safe moves
                if game_state.board.get_cell_state(new_pos) == 0:
                    neighbors.append(new_pos)
            return neighbors
        
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for neighbor in get_safe_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No safe path found
    
    def find_any_safe_move(self, game_state, my_agent, opponent):
        """Find any safe move as last resort"""
        my_pos = my_agent.trail[-1]
        safe_directions = []
        
        for direction in range(4):  # RIGHT, LEFT, DOWN, UP
            dx, dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][direction]
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == opponent.trail[-1]:
                safe_directions.append(direction)
        
        if safe_directions:
            return safe_directions[0]  # Return first safe direction
        
        # No safe moves - choose randomly and hope for the best
        return random.randint(0, 3)
    
    # COMPATIBLE METHODS with SimpleDQNAgent interface
    def get_state_representation(self, game_state, my_agent, opponent):
        """Original state representation (keep it simple)"""
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
        """Get valid action indices"""
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
            if direction == forbidden_direction:
                continue

            dx, dy = direction.value
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)

            is_safe = (game_state.board.get_cell_state(next_pos) == 0) or (next_pos == opponent.trail[-1])

            if is_safe:
                valid_actions.append(dir_idx * 2 + 0)

                if boosts_remaining > 0:
                    boost_pos = ((next_pos[0] + dx) % 20, (next_pos[1] + dy) % 18)
                    boost_safe = (game_state.board.get_cell_state(boost_pos) == 0) or (boost_pos == opponent.trail[-1])

                    if boost_safe:
                        valid_actions.append(dir_idx * 2 + 1)

        if not valid_actions:
            for dir_idx, direction in enumerate(directions):
                if direction != forbidden_direction:
                    valid_actions.append(dir_idx * 2 + 0)

        return valid_actions
    
    def smart_act(self, state, valid_actions, game_state, my_agent, opponent):
        """Smart action selection with emergency override"""
        # Check if we're in immediate danger
        in_danger, safe_moves = self.is_in_danger(game_state, my_agent, opponent)
        
        # EMERGENCY OVERRIDE: Use pathfinding to escape danger
        if in_danger:
            print(f"🚨 EMERGENCY: Only {safe_moves} safe moves left! Using pathfinding...")
            emergency_direction = self.find_emergency_escape(game_state, my_agent, opponent)
            
            # Convert direction to action (prefer non-boost in emergency)
            emergency_action = emergency_direction * 2  # Non-boost
            
            if emergency_action in valid_actions:
                self.last_emergency_action = emergency_action
                return emergency_action
            else:
                # Try boost version
                emergency_action_boost = emergency_direction * 2 + 1
                if emergency_action_boost in valid_actions:
                    self.last_emergency_action = emergency_action_boost
                    return emergency_action_boost
        
        # NORMAL OPERATION: Use DQN for strategic decisions
        if np.random.random() <= self.epsilon:
            # Exploration
            non_boost_actions = [a for a in valid_actions if a % 2 == 0]
            if non_boost_actions and (np.random.random() < 0.9 or not valid_actions):
                return random.choice(non_boost_actions)
            return random.choice(valid_actions) if valid_actions else 0
        else:
            # Exploitation with DQN
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                q_values = q_values.cpu().numpy()[0]
            
            # Only consider valid actions
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action

    # For compatibility with training scripts that call act() directly
    def act(self, state, valid_actions):
        """Default act method for compatibility - uses smart_act with dummy parameters"""
        # This is a fallback - smart_act should be called with full parameters
        if not valid_actions:
            return 0
        
        if np.random.random() <= self.epsilon:
            non_boost_actions = [a for a in valid_actions if a % 2 == 0]
            if non_boost_actions and (np.random.random() < 0.9 or not valid_actions):
                return random.choice(non_boost_actions)
            return random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
                q_values = q_values.cpu().numpy()[0]
            
            valid_q_values = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q_values, key=lambda x: x[1])[0]
            return best_action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_reward(self, my_agent, opponent, prev_my_alive, prev_opp_alive, game_state, action_used):
        """Reward function with emergency survival bonus"""
        reward = 0
        
        # Massive game outcome rewards
        if not my_agent.alive and prev_my_alive:
            return -100.0
        
        if not opponent.alive and prev_opp_alive:
            return 100.0
        
        # Base survival reward
        reward += 0.1
        
        # Emergency survival bonus
        in_danger, safe_moves = self.is_in_danger(game_state, my_agent, opponent)
        if in_danger and my_agent.alive:
            reward += 2.0  # Bonus for surviving dangerous situation
        
        # Safety assessment
        my_pos = my_agent.trail[-1]
        safe_moves_count = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == opponent.trail[-1]:
                safe_moves_count += 1
        
        # Safety rewards/penalties
        if safe_moves_count >= 3:
            reward += 0.3
        elif safe_moves_count == 2:
            reward += 0.1
        elif safe_moves_count == 1:
            reward -= 0.5
        else:
            reward -= 2.0
        
        return reward

    # Alias for compatibility
    get_enhanced_reward = get_reward
    
    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\emergency_dqn_model.pth")
        print(f"Emergency DQN Model saved (epsilon: {self.epsilon:.3f})")
    
    def load_model(self):
        try:
            checkpoint = torch.load(r"C:\Users\Phillip\Desktop\coding projects\TamuDataton2025\Datathon2025\case_closed\case-closed-starter-code-main\case-closed-starter-code-main\emergency_dqn_model.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Emergency DQN Model loaded (epsilon: {self.epsilon:.3f})")
        except FileNotFoundError:
            print("No emergency model found, starting fresh")

    def new_episode(self):
        """Optional method for episode reset - for compatibility"""
        pass