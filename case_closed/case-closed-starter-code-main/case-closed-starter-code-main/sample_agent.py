import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
import random
import pickle
from collections import defaultdict

from case_closed_game import Game, Direction, GameResult, EMPTY

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(8))  # 4 directions * 2 boost states
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.last_state = None
        self.last_action = None
        
        # Load existing Q-table if available
        self.load_q_table()
    
    def get_state_representation(self, game_state, my_agent, opponent):
        """Convert game state to a simplified representation for Q-learning"""
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        # Basic state features
        state_features = []
        
        # 1. Danger in each direction (0 = safe, 1 = danger)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # RIGHT, LEFT, DOWN, UP
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) != 0 and next_pos != opp_pos:
                state_features.append(1)  # Danger
            else:
                state_features.append(0)  # Safe
        
        # 2. Relative position to opponent
        dx = (opp_pos[0] - my_pos[0]) % 20
        if dx > 10: dx = dx - 20
        dy = (opp_pos[1] - my_pos[1]) % 18
        if dy > 9: dy = dy - 18
        
        # 3. Distance to opponent (discretized)
        distance = abs(dx) + abs(dy)
        if distance < 3:
            state_features.append(0)  # Very close
        elif distance < 6:
            state_features.append(1)  # Close
        elif distance < 10:
            state_features.append(2)  # Medium
        else:
            state_features.append(3)  # Far
        
        # 4. Direction to opponent
        if abs(dx) > abs(dy):
            if dx > 0:
                state_features.append(0)  # Right
            else:
                state_features.append(1)  # Left
        else:
            if dy > 0:
                state_features.append(2)  # Down
            else:
                state_features.append(3)  # Up
        
        # 5. Boost available
        state_features.append(1 if my_agent.boosts_remaining > 0 else 0)
        
        return tuple(state_features)
    
    def choose_action(self, state, boosts_remaining):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: random action
            direction = random.choice([0, 1, 2, 3])  # RIGHT, LEFT, DOWN, UP
            use_boost = random.choice([0, 1]) if boosts_remaining > 0 else 0
        else:
            # Exploit: best action from Q-table
            q_values = self.q_table[state]
            action_index = np.argmax(q_values)
            direction = action_index // 2
            use_boost = action_index % 2
        
        return direction, use_boost
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning formula"""
        if self.last_state is not None and self.last_action is not None:
            current_q = self.q_table[self.last_state][self.last_action]
            
            if done:
                next_max = 0
            else:
                next_max = np.max(self.q_table[next_state])
            
            # Q-learning update
            new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
            self.q_table[self.last_state][self.last_action] = new_q
        
        # Store current state and action for next update
        if not done:
            self.last_state = state
            self.last_action = action
        else:
            self.last_state = None
            self.last_action = None
    
    def get_reward(self, my_agent, opponent, prev_my_alive, prev_opp_alive, game_state):
        """Calculate reward based on game outcome and situation"""
        reward = 0
        
        # Survival reward
        if my_agent.alive:
            reward += 1
        
        # Death penalty
        if not my_agent.alive and prev_my_alive:
            reward -= 100
        
        # Opponent death reward
        if not opponent.alive and prev_opp_alive:
            reward += 50
        
        # Distance to opponent (closer is better for aggressive play)
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        dx = (opp_pos[0] - my_pos[0]) % 20
        if dx > 10: dx = dx - 20
        dy = (opp_pos[1] - my_pos[1]) % 18
        if dy > 9: dy = dy - 18
        distance = abs(dx) + abs(dy)
        
        # Reward for closing distance
        reward += (20 - distance) * 0.1
        
        # Penalty for being trapped
        safe_moves = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game_state.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
                safe_moves += 1
        
        if safe_moves == 0:
            reward -= 10
        elif safe_moves == 1:
            reward -= 5
        
        return reward
    
    def save_q_table(self):
        """Save Q-table to file"""
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_q_table(self):
        """Load Q-table from file"""
        try:
            with open('q_table.pkl', 'rb') as f:
                loaded_table = pickle.load(f)
                self.q_table.update(loaded_table)
            print("Loaded existing Q-table")
        except FileNotFoundError:
            print("No existing Q-table found, starting fresh")
    
    def decay_epsilon(self, episode):
        """Decay epsilon over time to reduce exploration"""
        self.epsilon = max(0.01, 0.1 * (0.99 ** episode))

# Global Q-learning agent
q_agent = QLearningAgent()

def simple_heuristic_agent(game, my_agent, opponent):
    """Simple heuristic agent for training"""
    my_pos = my_agent.trail[-1]
    opp_pos = opponent.trail[-1]
    
    # Calculate direction to opponent
    dx = (opp_pos[0] - my_pos[0]) % 20
    if dx > 10: dx = dx - 20
    dy = (opp_pos[1] - my_pos[1]) % 18
    if dy > 9: dy = dy - 18
    
    # Prefer moves toward opponent that are safe
    safe_directions = []
    for i, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
        if game.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
            safe_directions.append(i)
    
    if safe_directions:
        # Choose direction that minimizes distance to opponent
        best_dir = None
        best_distance = float('inf')
        for dir_idx in safe_directions:
            test_dx, test_dy = [(1, 0), (-1, 0), (0, 1), (0, -1)][dir_idx]
            test_pos = ((my_pos[0] + test_dx) % 20, (my_pos[1] + test_dy) % 18)
            dist = abs((opp_pos[0] - test_pos[0]) % 20) + abs((opp_pos[1] - test_pos[1]) % 18)
            if dist < best_distance:
                best_distance = dist
                best_dir = dir_idx
        return best_dir, 0  # Don't use boost for training opponent
    
    # Fallback: random safe move
    return random.randint(0, 3), 0

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server. 

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


def rl_send_move():
    """RL-based move selection"""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opponent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        boosts_remaining = my_agent.boosts_remaining
        
        # Store previous state for reward calculation
        prev_my_alive = my_agent.alive
        prev_opp_alive = opponent.alive
    
    # Get state representation
    current_state = q_agent.get_state_representation(GLOBAL_GAME, my_agent, opponent)
    
    # Choose action
    direction_index, use_boost = q_agent.choose_action(current_state, boosts_remaining)
    
    # Convert direction index to string
    direction_map = {0: "RIGHT", 1: "LEFT", 2: "DOWN", 3: "UP"}
    move = direction_map[direction_index]
    
    # Add boost if chosen and available
    if use_boost and boosts_remaining > 0:
        move += ":BOOST"
    
    # Calculate action index for Q-table (0-7)
    action_index = direction_index * 2 + use_boost
    
    # Calculate reward based on previous action
    if q_agent.last_state is not None:
        reward = q_agent.get_reward(my_agent, opponent, prev_my_alive, prev_opp_alive, GLOBAL_GAME)
        q_agent.update_q_value(current_state, action_index, reward, current_state, False)
    
    print(f'RL Agent - State: {current_state}, Action: {move}, Boost used: {use_boost}')
    
    return jsonify({"move": move}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Use RL agent for move selection"""
    return rl_send_move()


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    
    # Save Q-table at the end of each game
    q_agent.save_q_table()
    
    return jsonify({"status": "acknowledged"}), 200


# Training function (run this separately to train the agent)
def train_rl_agent(episodes=1000):
    """Train the RL agent by playing against itself or simple opponent"""
    for episode in range(episodes):
        game = Game()
        q_agent.decay_epsilon(episode)
        
        # Reset agent state
        q_agent.last_state = None
        q_agent.last_action = None
        
        prev_agent1_alive = True
        prev_agent2_alive = True
        
        while game.agent1.alive and game.agent2.alive and game.turns < 200:
            # Agent 1 move (RL)
            state1 = q_agent.get_state_representation(game, game.agent1, game.agent2)
            dir_idx1, boost1 = q_agent.choose_action(state1, game.agent1.boosts_remaining)
            direction1 = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP][dir_idx1]
            action_idx1 = dir_idx1 * 2 + boost1
            
            # Agent 2 move (simple heuristic)
            dir_idx2, boost2 = simple_heuristic_agent(game, game.agent2, game.agent1)
            direction2 = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP][dir_idx2]
            
            # Store state before move for reward calculation
            prev_agent1_alive = game.agent1.alive
            prev_agent2_alive = game.agent2.alive
            
            # Execute moves
            game.step(direction1, direction2, boost1, boost2)
            
            # Update Q-values for agent 1
            next_state1 = q_agent.get_state_representation(game, game.agent1, game.agent2)
            reward1 = q_agent.get_reward(game.agent1, game.agent2, prev_agent1_alive, prev_agent2_alive, game)
            done = not (game.agent1.alive and game.agent2.alive)
            q_agent.update_q_value(next_state1, action_idx1, reward1, next_state1, done)
        
        # Final update for terminal state
        if q_agent.last_state is not None:
            final_reward = q_agent.get_reward(game.agent1, game.agent2, True, True, game)
            q_agent.update_q_value(q_agent.last_state, q_agent.last_action, final_reward, None, True)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {q_agent.epsilon:.3f}")
            q_agent.save_q_table()
    
    q_agent.save_q_table()
    print("Training completed!")


if __name__ == "__main__":
    # Uncomment the line below to train the agent
    train_rl_agent(episodes=1000)
    
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)