from dqn_agent import SimpleDQNAgent
from case_closed_game import Game, Direction
import random
import numpy as np

def defensive_opponent(game, my_agent, opponent):
    """Opponent that runs away and plays safe"""
    my_pos = my_agent.trail[-1]
    opp_pos = opponent.trail[-1]
    
    current_direction = my_agent.direction
    reverse_directions = {
        Direction.RIGHT: Direction.LEFT,
        Direction.LEFT: Direction.RIGHT,
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP
    }
    forbidden_direction = reverse_directions.get(current_direction)
    
    # Score moves by safety and distance from opponent
    scored_moves = []
    directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
    
    for i, direction in enumerate(directions):
        if direction == forbidden_direction:
            continue
            
        dx, dy = direction.value
        next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
        
        # Check safety
        if game.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
            # Calculate new distance from opponent
            new_dx = (opp_pos[0] - next_pos[0]) % 20
            if new_dx > 10: new_dx = new_dx - 20
            new_dy = (opp_pos[1] - next_pos[1]) % 18
            if new_dy > 9: new_dy = new_dy - 18
            new_distance = abs(new_dx) + abs(new_dy)
            
            # Prefer larger distances (defensive)
            score = new_distance
            
            # Bonus for future safety
            future_safe = 0
            for dx2, dy2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                future_pos = ((next_pos[0] + dx2) % 20, (next_pos[1] + dy2) % 18)
                if game.board.get_cell_state(future_pos) == 0 or future_pos == opp_pos:
                    future_safe += 1
            score += future_safe * 5
            
            scored_moves.append((i, score))
    
    if scored_moves:
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return scored_moves[0][0], 0
    
    # Fallback
    valid_dirs = [i for i in range(4) if directions[i] != forbidden_direction]
    if valid_dirs:
        return random.choice(valid_dirs), 0
    else:
        dir_map = {Direction.RIGHT: 0, Direction.LEFT: 1, Direction.DOWN: 2, Direction.UP: 3}
        return dir_map[current_direction], 0

def aggressive_opponent(game, my_agent, opponent):
    """Opponent that chases and attacks"""
    my_pos = my_agent.trail[-1]
    opp_pos = opponent.trail[-1]
    
    current_direction = my_agent.direction
    reverse_directions = {
        Direction.RIGHT: Direction.LEFT,
        Direction.LEFT: Direction.RIGHT,
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP
    }
    forbidden_direction = reverse_directions.get(current_direction)
    
    # Calculate distance and direction to opponent
    dx = (opp_pos[0] - my_pos[0]) % 20
    if dx > 10: dx = dx - 20
    dy = (opp_pos[1] - my_pos[1]) % 18
    if dy > 9: dy = dy - 18
    
    # Prefer moves that get closer to opponent
    preferred_dirs = []
    if abs(dx) > abs(dy):
        if dx > 0:
            preferred_dirs.append(Direction.RIGHT)
        else:
            preferred_dirs.append(Direction.LEFT)
        if dy > 0:
            preferred_dirs.append(Direction.DOWN)
        else:
            preferred_dirs.append(Direction.UP)
    else:
        if dy > 0:
            preferred_dirs.append(Direction.DOWN)
        else:
            preferred_dirs.append(Direction.UP)
        if dx > 0:
            preferred_dirs.append(Direction.RIGHT)
        else:
            preferred_dirs.append(Direction.LEFT)
    
    # Try preferred directions first
    for direction in preferred_dirs:
        if direction != forbidden_direction:
            dx, dy = direction.value
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
                dir_map = {Direction.RIGHT: 0, Direction.LEFT: 1, Direction.DOWN: 2, Direction.UP: 3}
                return dir_map[direction], 0
    
    # Fallback to any safe move
    directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
    for direction in directions:
        if direction != forbidden_direction:
            dx, dy = direction.value
            next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
            if game.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
                dir_map = {Direction.RIGHT: 0, Direction.LEFT: 1, Direction.DOWN: 2, Direction.UP: 3}
                return dir_map[direction], 0
    
    # Last resort
    dir_map = {Direction.RIGHT: 0, Direction.LEFT: 1, Direction.DOWN: 2, Direction.UP: 3}
    return dir_map[current_direction], 0

def random_opponent(game, my_agent, opponent):
    """Opponent that makes random (but valid) moves"""
    current_direction = my_agent.direction
    reverse_directions = {
        Direction.RIGHT: Direction.LEFT,
        Direction.LEFT: Direction.RIGHT,
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP
    }
    forbidden_direction = reverse_directions.get(current_direction)
    
    my_pos = my_agent.trail[-1]
    opp_pos = opponent.trail[-1]
    
    # Get all valid moves
    valid_moves = []
    directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
    
    for i, direction in enumerate(directions):
        if direction == forbidden_direction:
            continue
            
        dx, dy = direction.value
        next_pos = ((my_pos[0] + dx) % 20, (my_pos[1] + dy) % 18)
        if game.board.get_cell_state(next_pos) == 0 or next_pos == opp_pos:
            valid_moves.append(i)
    
    if valid_moves:
        return random.choice(valid_moves), 0
    else:
        dir_map = {Direction.RIGHT: 0, Direction.LEFT: 1, Direction.DOWN: 2, Direction.UP: 3}
        return dir_map[current_direction], 0

def multiple_opponents(game, my_agent, opponent, episode):
    """Rotate between different opponent strategies"""
    if episode % 3 == 0:
        return defensive_opponent(game, my_agent, opponent)
    elif episode % 3 == 1:
        return aggressive_opponent(game, my_agent, opponent)
    else:
        return random_opponent(game, my_agent, opponent)

def train_advanced(episodes=2000, save_interval=100):
    """Train against multiple opponent strategies"""
    
    print("Initializing Advanced DQN Training...")
    agent = SimpleDQNAgent(state_size=15, action_size=8)
    
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    
    print(f"Starting advanced training for {episodes} episodes...")
    print("Opponents: Defensive (33%) | Aggressive (33%) | Random (33%)")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    
    for episode in range(episodes):
        game = Game()
        episode_reward = 0
        
        while game.agent1.alive and game.agent2.alive and game.turns < 200:
            # RL Agent (Agent 1)
            state = agent.get_state_representation(game, game.agent1, game.agent2)
            # get_valid_actions expects (game_state, my_agent, boosts_remaining, opponent)
            valid_actions = agent.get_valid_actions(game, game.agent1, game.agent1.boosts_remaining, game.agent2)
            action = agent.act(state, valid_actions)
            
            # Convert action
            direction_idx = action // 2
            use_boost = action % 2
            direction = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP][direction_idx]
            action_used = (direction_idx, use_boost)
            
            # Opponent (rotating strategy)
            opp_dir_idx, opp_boost = multiple_opponents(game, game.agent2, game.agent1, episode)
            opp_direction = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP][opp_dir_idx]
            
            # Store previous state
            prev_agent1_alive = game.agent1.alive
            prev_agent2_alive = game.agent2.alive
            
            # Execute moves
            game.step(direction, opp_direction, use_boost, opp_boost)
            
            # Get next state and reward
            next_state = agent.get_state_representation(game, game.agent1, game.agent2)
            reward = agent.get_reward(game.agent1, game.agent2, prev_agent1_alive, prev_agent2_alive, game, action_used)
            done = not (game.agent1.alive and game.agent2.alive)
            
            # Remember and replay
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            episode_reward += reward
        
        # Track results
        if not game.agent1.alive and not game.agent2.alive:
            draws += 1
            result = "DRAW"
        elif not game.agent1.alive:
            losses += 1
            result = "LOSS"
        else:
            wins += 1
            result = "WIN"
        
        total_reward += episode_reward
        
        # Print progress
        if (episode + 1) % save_interval == 0 or (episode + 1) <= 10:
            win_rate = wins / (episode + 1) * 100
            avg_reward = total_reward / (episode + 1)
            
            opponent_type = ["DEFENSIVE", "AGGRESSIVE", "RANDOM"][episode % 3]
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Opponent: {opponent_type} | "
                  f"Result: {result} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
            
            agent.save_model()
    
    agent.save_model()
    final_win_rate = wins / episodes * 100
    print(f"\n=== Advanced Training Complete ===")
    print(f"Final Win Rate: {final_win_rate:.1f}%")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print("Agent trained against multiple strategies!")

if __name__ == "__main__":
    train_advanced(episodes=2000, save_interval=100)