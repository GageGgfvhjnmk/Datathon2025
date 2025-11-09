"""
Heuristic-Guided DQN Agent for Tron Game
Solution 2: RL learns to pick best move from heuristic's top-K suggestions

Architecture:
- Heuristic evaluates all moves and ranks them
- DQN network learns Q-values for top-K candidates
- Epsilon-greedy exploration within top-K
- Fast training via guided exploration

HARD-CODED SAFETY FEATURES:
- ✅ Never moves backwards (filtered)
- ✅ Never crosses own trail (HARD-CODED - not learned!)
- ✅ Only learns strategy: when to boost, when to attack/defend
- ✅ Dramatically reduces training time by avoiding -25 penalty loops

This means the RL agent will ONLY learn:
1. When to use boost
2. Whether to move toward or away from opponent
3. How to claim territory efficiently
4. When to take risks vs play safe

It will NEVER waste training time learning "don't hit yourself"!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from case_closed_game import Direction
import agent_strategies
import heapq

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for move selection
    Input: Board state + candidate move features
    Output: Q-value for each of top-K candidate moves
    """
    def __init__(self, board_width=18, board_height=20, feature_dim=32, max_candidates=5):
        super(DQNNetwork, self).__init__()
        
        self.board_width = board_width
        self.board_height = board_height
        self.max_candidates = max_candidates
        
        # Convolutional layers for board state
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 channels: empty, agent1, agent2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = board_width * board_height * 64
        
        # Fully connected layers for board representation
        self.fc_board = nn.Linear(conv_output_size, 256)
        
        # Feature processing for candidate moves
        # Features: [heuristic_score, area, exits, territory_diff, mobility_ratio, etc.]
        self.fc_features = nn.Linear(feature_dim, 128)
        
        # Combined processing
        self.fc_combined = nn.Linear(256 + 128, 128)
        self.fc_output = nn.Linear(128, 1)  # Q-value for this candidate
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, board_state, candidate_features):
        """
        Forward pass
        board_state: (batch, 3, height, width) - tensor representation of board
        candidate_features: (batch, feature_dim) - features for one candidate move
        Returns: (batch, 1) - Q-value
        """
        # Process board through convolutions
        x = self.relu(self.conv1(board_state))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc_board(x))
        x = self.dropout(x)
        
        # Process candidate features
        f = self.relu(self.fc_features(candidate_features))
        
        # Combine
        combined = torch.cat([x, f], dim=1)
        combined = self.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        q_value = self.fc_output(combined)
        
        return q_value


class HeuristicGuidedDQNAgent:
    """
    RL Agent that learns to select from heuristic's top-K candidates
    """
    def __init__(self, 
                 board_width=18, 
                 board_height=20,
                 top_k=5,
                 learning_rate=0.0005,
                 gamma=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=0.995,
                 memory_size=10000,
                 batch_size=64,
                 target_update_freq=10,
                 device=None):
        
        self.board_width = board_width
        self.board_height = board_height
        self.top_k = top_k
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"RL Agent using device: {self.device}")
        
        # Networks
        self.policy_net = DQNNetwork(board_width, board_height, feature_dim=32, max_candidates=top_k).to(self.device)
        self.target_net = DQNNetwork(board_width, board_height, feature_dim=32, max_candidates=top_k).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.total_reward = 0
        self.training_mode = True
        
        # Emergency pathfinding tracking
        self.last_emergency_action = None
        
    def board_to_tensor(self, board, agent1_trail, agent2_trail):
        """
        Convert board to tensor representation
        Returns: (3, height, width) tensor
        Channel 0: Empty cells (1) vs occupied (0)
        Channel 1: Agent 1 trail
        Channel 2: Agent 2 trail
        """
        height = len(board)
        width = len(board[0])
        
        tensor = np.zeros((3, height, width), dtype=np.float32)
        
        # Channel 0: Empty cells
        for y in range(height):
            for x in range(width):
                if board[y][x] == 0:
                    tensor[0, y, x] = 1.0
        
        # Channel 1: Agent 1 trail
        for x, y in agent1_trail:
            tensor[1, y, x] = 1.0
        
        # Channel 2: Agent 2 trail
        for x, y in agent2_trail:
            tensor[2, y, x] = 1.0
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
    
    def extract_candidate_features(self, move_eval):
        """
        Extract features from move evaluation
        Returns: feature vector for neural network
        """
        features = [
            move_eval.get('area', 0) / 360.0,  # Normalize by board size
            move_eval.get('avg_depth', 0) / 20.0,
            move_eval.get('immediate_exits', 0) / 4.0,
            move_eval.get('dead_ends', 0) / 10.0,
            move_eval.get('choke_points', 0) / 10.0,
            move_eval.get('max_depth', 0) / 50.0,
            move_eval.get('opp_area', 0) / 360.0,
            1.0 if move_eval.get('can_trap_opp', False) else 0.0,
            1.0 if move_eval.get('can_be_trapped', False) else 0.0,
            move_eval.get('my_territory', 0) / 360.0,
            move_eval.get('opp_territory', 0) / 360.0,
            move_eval.get('contested', 0) / 360.0,
            move_eval.get('critical_captured', 0) / 10.0,
            move_eval.get('manhattan_dist', 0) / 40.0,
            move_eval.get('corridor_risk', 0) / 5.0,
            move_eval.get('mobility_ratio', 1.0),
            move_eval.get('opp_mobility', 0) / 4.0,
            move_eval.get('my_mobility', 0) / 4.0,
            move_eval.get('compactness', 0),
            1.0 if move_eval.get('boost', False) else 0.0,
        ]
        
        # Pad to 32 features
        while len(features) < 32:
            features.append(0.0)
        
        return torch.FloatTensor(features[:32]).unsqueeze(0).to(self.device)
    
    def get_top_k_from_heuristic(self, game, player_number):
        """
        Use heuristic to get top-K candidate moves
        Returns: list of (direction, boost, evaluation_dict)
        """
        # Build state for heuristic
        state = {
            "board": game.board.grid,
            "agent1_trail": list(game.agent1.trail),
            "agent2_trail": list(game.agent2.trail),
            "agent1_boosts": game.agent1.boosts_remaining,
            "agent2_boosts": game.agent2.boosts_remaining,
            "agent1_direction": game.agent1.direction,
            "agent2_direction": game.agent2.direction,
            "player_number": player_number
        }
        
        # DEBUG: Check what type board is
        print(f"DEBUG: type(game.board.grid) = {type(game.board.grid)}")
        if hasattr(game.board.grid, '__len__') and len(game.board.grid) > 0:
            print(f"DEBUG: type(game.board.grid[0]) = {type(game.board.grid[0])}")
            if hasattr(game.board.grid[0], '__len__') and len(game.board.grid[0]) > 0:
                print(f"DEBUG: type(game.board.grid[0][0]) = {type(game.board.grid[0][0])}")
                print(f"DEBUG: sample board values: grid[0][0]={game.board.grid[0][0]}, grid[0][1]={game.board.grid[0][1]}, grid[1][0]={game.board.grid[1][0]}")
        
        # Get all move evaluations from heuristic
        from agent_strategies import pick_move
        
        # We need to extract the move evaluations, not just the best move
        # For now, call the heuristic's internal evaluation
        # This is a simplified version - you may want to refactor agent_strategies
        # to expose the move_evaluations list
        
        # Temporary: generate candidates by calling heuristic evaluation logic
        candidates = self._get_heuristic_candidates(state)
        
        # Sort by heuristic score and take top-K
        candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score
        
        return candidates[:self.top_k]
    
    def _check_if_trapped(self, board, new_pos, my_trail_set, my_player_id, min_space=15):
        """
        Check if a move creates a negative space that traps the agent
        Returns: (is_trapped, available_space)
        
        HARD-CODED SAFETY: Detects if moving to new_pos creates an enclosed area too small to survive
        """
        width, height = len(board[0]), len(board)
        dir_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        # Flood fill from new position to count available empty space
        visited = set()
        queue = [new_pos]
        visited.add(new_pos)
        space_count = 0
        
        # Include our new position in the trail set temporarily for this check
        temp_trail_set = my_trail_set | {new_pos}
        
        while queue and space_count <= min_space + 5:  # Stop early if we find enough space
            cx, cy = queue.pop(0)
            space_count += 1
            
            for dx, dy in dir_map:
                nx = (cx + dx) % width
                ny = (cy + dy) % height
                
                if (nx, ny) not in visited:
                    # Check if cell is free (not any wall/trail)
                    cell_value = board[ny][nx]
                    if cell_value == 0 and (nx, ny) not in temp_trail_set:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        # If available space is less than minimum, we're trapped!
        is_trapped = space_count < min_space
        return is_trapped, space_count
    
    def _get_heuristic_candidates(self, state):
        """
        Helper to get candidate moves from heuristic evaluation
        Uses the actual heuristic from agent_strategies for proper evaluation
        HARD-CODED SAFETY: 
        - Filters out any move that would hit own trail
        - Filters out moves that create negative space traps (NEW!)
        """
        from agent_strategies import pick_move
        import agent_strategies as ast
        
        DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
        board = state["board"]
        player = state["player_number"]
        boosts = state[f"agent{player}_boosts"]
        current_direction = state[f"agent{player}_direction"]
        
        # Get backwards direction (not allowed)
        cur_dx, cur_dy = current_direction.value
        backwards_direction = None
        dir_map = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
        
        for dir_name, (dx, dy) in dir_map.items():
            if (dx, dy) == (-cur_dx, -cur_dy):
                backwards_direction = dir_name
                break
        
        candidates = []
        
        # Get agent positions
        agent1_trail = state["agent1_trail"]
        agent2_trail = state["agent2_trail"]
        my_trail = agent1_trail if player == 1 else agent2_trail
        opp_trail = agent2_trail if player == 1 else agent1_trail
        my_pos = tuple(my_trail[-1])
        
        # Convert trail to set for O(1) lookup
        my_trail_set = set(tuple(pos) for pos in my_trail)
        
        # Get my player ID on the board (1 or 2)
        my_player_id = player
        
        # DEBUG: Print board state
        print(f"DEBUG: Player {player} position: {my_pos}")
        print(f"DEBUG: Board dimensions: {len(board[0])}x{len(board)}")
        print(f"DEBUG: My trail length: {len(my_trail_set)}")
        # Print actual board values around player
        x, y = my_pos
        width, height = len(board[0]), len(board)
        print(f"DEBUG: Board values around player at ({x},{y}):")
        print(f"  UP ({x},{(y-1)%height}): {board[(y-1)%height][x]}")
        print(f"  DOWN ({x},{(y+1)%height}): {board[(y+1)%height][x]}")
        print(f"  LEFT ({(x-1)%width},{y}): {board[y][(x-1)%width]}")
        print(f"  RIGHT ({(x+1)%width},{y}): {board[y][(x+1)%width]}")
        
        # Simple evaluation for each valid direction
        for direction in DIRECTIONS:
            # Skip backwards
            if direction == backwards_direction:
                continue
                
            for use_boost in ([False, True] if boosts > 0 else [False]):
                # Check if move is valid
                dx, dy = dir_map[direction]
                width, height = len(board[0]), len(board)
                
                # Simulate move and CHECK FOR SELF-COLLISION (HARD-CODED SAFETY)
                x, y = my_pos
                steps = 2 if use_boost else 1
                valid = True
                hits_own_trail = False
                
                for step in range(steps):
                    x = (x + dx) % width
                    y = (y + dy) % height
                    
                    # CRITICAL SAFETY CHECK: Would we hit OUR OWN trail?
                    # Note: board uses AGENT=1 for all trails regardless of player
                    # So we must check against our trail set, not board cell value
                    
                    # Check if position is in our trail set (HARD-CODED SAFETY)
                    if (x, y) in my_trail_set:
                        hits_own_trail = True
                        valid = False
                        print(f"DEBUG: {direction} (boost={use_boost}) step {step+1}: hits own trail at ({x},{y})")
                        break
                    
                    # Check if we hit any trail (wall) - either player
                    # AGENT = 1 means occupied by any trail
                    cell_value = board[y][x]
                    print(f"DEBUG: {direction} (boost={use_boost}) step {step+1}: checking ({x},{y}), cell_value={cell_value} (numeric)")
                    if cell_value != 0:  # 0 = EMPTY, 1 = AGENT trail
                        valid = False
                        print(f"DEBUG: {direction} (boost={use_boost}) step {step+1}: hits occupied at ({x},{y}), cell={cell_value}")
                        break
                
                # HARD-CODED FILTER: Skip moves that would cause self-collision
                if not valid or hits_own_trail:
                    print(f"DEBUG: Skipping {direction} (boost={use_boost}): valid={valid}, hits_own={hits_own_trail}")
                    continue
                
                # NEW HARD-CODED SAFETY: Check if move creates a trap (negative space)
                new_pos = (x, y)
                is_trapped, available_space = self._check_if_trapped(
                    board, new_pos, my_trail_set, my_player_id, min_space=15
                )
                
                if is_trapped:
                    # This move would trap us in a small space - SKIP IT!
                    print(f"DEBUG: {direction} (boost={use_boost}) creates trap: space={available_space}")
                    continue
                
                print(f"DEBUG: {direction} (boost={use_boost}) is VALID! space={available_space}")
                
                # Create evaluation dict with realistic values
                new_pos = (x, y)
                
                # Count immediate exits from new position
                immediate_exits = 0
                for edx, edy in dir_map.values():
                    nx = (new_pos[0] + edx) % width
                    ny = (new_pos[1] + edy) % height
                    if board[ny][nx] == 0:
                        immediate_exits += 1
                
                # Simple flood fill for area
                visited = set()
                queue = [new_pos]
                visited.add(new_pos)
                area = 0
                
                while queue and area < 100:  # Limit search
                    cx, cy = queue.pop(0)
                    area += 1
                    
                    for edx, edy in dir_map.values():
                        nx = (cx + edx) % width
                        ny = (cy + edy) % height
                        if board[ny][nx] == 0 and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                eval_dict = {
                    'direction': direction,
                    'boost': use_boost,
                    'area': area,
                    'available_space': available_space,  # NEW: space after this move
                    'creates_trap': False,  # Already filtered out trapped moves
                    'immediate_exits': immediate_exits,
                    'avg_depth': area / max(immediate_exits, 1),
                    'dead_ends': 1 if immediate_exits <= 1 else 0,
                    'choke_points': 1 if immediate_exits == 2 else 0,
                    'max_depth': min(area, 50),
                    'opp_area': 100,  # Placeholder
                    'can_trap_opp': False,
                    'can_be_trapped': immediate_exits <= 2,
                    'my_territory': area,
                    'opp_territory': 100,
                    'contested': 50,
                    'critical_captured': 0,
                    'manhattan_dist': 10,
                    'corridor_risk': 0,
                    'mobility_ratio': 1.0,
                    'opp_mobility': 3,
                    'my_mobility': immediate_exits,
                    'compactness': 0.5,
                }
                
                # Simple heuristic score - bonus for more space!
                score = area * 2 + immediate_exits * 10 + available_space * 0.5
                if use_boost:
                    score -= 5  # Small penalty for using boost
                
                candidates.append((direction, use_boost, score, eval_dict))
        
        print(f"DEBUG: Generated {len(candidates)} candidates")
        return candidates
    
    def select_action(self, game, player_number, training=None):
        """
        Select action using epsilon-greedy policy over top-K heuristic candidates
        HARD-CODED SAFETY: All candidates are pre-filtered to never hit own trail
        """
        if training is None:
            training = self.training_mode
        
        # Get top-K candidates from heuristic (already filtered for self-collision)
        candidates = self.get_top_k_from_heuristic(game, player_number)
        
        if not candidates:
            # EMERGENCY FALLBACK: No safe moves available - we're trapped!
            # Try to find ANY move that doesn't hit own trail
            print("⚠️ WARNING: No safe candidates found! Emergency fallback...")
            
            my_trail = list(game.agent2.trail) if player_number == 2 else list(game.agent1.trail)
            my_trail_set = set(tuple(pos) for pos in my_trail)
            my_pos = tuple(my_trail[-1])
            prev_pos = tuple(my_trail[-2]) if len(my_trail) >= 2 else None
            
            dir_map = {
                "UP": (0, -1, Direction.UP),
                "DOWN": (0, 1, Direction.DOWN),
                "LEFT": (-1, 0, Direction.LEFT),
                "RIGHT": (1, 0, Direction.RIGHT)
            }
            
            width, height = game.board.width, game.board.height
            
            # Try each direction - pick first one that doesn't hit own trail AND isn't backwards
            for dir_name, (dx, dy, dir_enum) in dir_map.items():
                nx = (my_pos[0] + dx) % width
                ny = (my_pos[1] + dy) % height
                
                # Check if this is backwards (going to previous position)
                if prev_pos and (nx, ny) == prev_pos:
                    continue  # Backwards movement - skip it
                
                # Check if this cell is in our own trail (using trail set)
                # Note: board uses AGENT=1 for all trails, so we must check against our trail set
                if (nx, ny) in my_trail_set:
                    continue  # Definitely our trail - skip it
                
                # Found a move that doesn't hit our trail and isn't backwards!
                # (might hit opponent trail or be empty, both are better than self-collision)
                print(f"  → Emergency: Picking {dir_name} (avoids self-collision)")
                return dir_enum, False
            
            # Truly trapped - all directions hit our own trail or go backwards
            # This should never happen with the hard-coded filter, but if it does,
            # just return UP (we're going to crash no matter what)
            print("  → Truly trapped! All directions hit own trail or go backwards.")
            return Direction.UP, False
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Explore: random candidate
            direction_str, use_boost, _, _ = random.choice(candidates)
        else:
            # Exploit: use DQN to pick best candidate
            board_tensor = self.board_to_tensor(
                game.board.grid,
                list(game.agent1.trail),
                list(game.agent2.trail)
            )
            
            best_q = -float('inf')
            best_candidate = candidates[0]
            
            with torch.no_grad():
                for direction_str, use_boost, heur_score, eval_dict in candidates:
                    features = self.extract_candidate_features(eval_dict)
                    q_value = self.policy_net(board_tensor, features)
                    
                    if q_value.item() > best_q:
                        best_q = q_value.item()
                        best_candidate = (direction_str, use_boost, heur_score, eval_dict)
            
            direction_str, use_boost, _, _ = best_candidate
        
        # Convert string direction to Direction enum
        direction_map = {
            "UP": Direction.UP,
            "DOWN": Direction.DOWN,
            "LEFT": Direction.LEFT,
            "RIGHT": Direction.RIGHT
        }
        
        return direction_map.get(direction_str, Direction.UP), use_boost
    
    # ==================== EMERGENCY PATHFINDING METHODS ====================
    # Added from emergency_dqn_agent.py for safety
    
    def is_in_danger(self, game, my_agent, opponent_agent, threshold=1):
        """
        Check if agent is in immediate danger (very few safe moves available)
        Returns True if safe_moves <= threshold
        """
        safe_moves = 0
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            if self._is_safe_move(game, my_agent, direction, use_boost=False):
                safe_moves += 1
        return safe_moves <= threshold
    
    def _is_safe_move(self, game, my_agent, direction, use_boost=False):
        """Check if a move is safe (doesn't immediately kill us)"""
        dx, dy = direction.value
        x, y = my_agent.trail[-1]
        width, height = len(game.board.grid[0]), len(game.board.grid)
        
        steps = 2 if use_boost else 1
        for _ in range(steps):
            x = (x + dx) % width
            y = (y + dy) % height
            if game.board.grid[y][x] != 0:
                return False
        return True
    
    def find_emergency_escape(self, game, my_agent, opponent_agent):
        """
        Emergency A* pathfinding to find safest area on the board
        Returns the first direction to move toward the safest area
        """
        # Find the safest area on the board
        safest_pos = self._find_safest_area(game, my_agent)
        if safest_pos is None:
            return None
        
        # Use A* to find path to safest area
        path = self._emergency_a_star(game, my_agent.trail[-1], safest_pos)
        
        if path and len(path) > 1:
            # Return direction to first step in path
            current_x, current_y = my_agent.trail[-1]
            next_x, next_y = path[1]
            width = len(game.board.grid[0])
            
            # Handle torus wrapping
            dx = next_x - current_x
            dy = next_y - current_y
            
            if abs(dx) > width // 2:
                dx = -dx / abs(dx) * (width - abs(dx))
            if abs(dy) > len(game.board.grid) // 2:
                dy = -dy / abs(dy) * (len(game.board.grid) - abs(dy))
            
            # Convert to Direction
            if abs(dx) > abs(dy):
                return Direction.RIGHT if dx > 0 else Direction.LEFT
            else:
                return Direction.DOWN if dy > 0 else Direction.UP
        
        return None
    
    def _find_safest_area(self, game, my_agent):
        """Find the position with the most empty space around it"""
        max_space = 0
        safest_pos = None
        
        width, height = len(game.board.grid[0]), len(game.board.grid)
        
        # Sample positions across the board
        for y in range(0, height, 3):
            for x in range(0, width, 3):
                if game.board.grid[y][x] == 0:
                    space = self._count_empty_space(game, (x, y), max_count=50)
                    if space > max_space:
                        max_space = space
                        safest_pos = (x, y)
        
        return safest_pos
    
    def _count_empty_space(self, game, start_pos, max_count=100):
        """Count empty spaces reachable from start_pos using flood fill"""
        visited = set()
        queue = deque([start_pos])
        count = 0
        width, height = len(game.board.grid[0]), len(game.board.grid)
        
        while queue and count < max_count:
            x, y = queue.popleft()
            
            if (x, y) in visited:
                continue
            
            if game.board.grid[y][x] != 0:
                continue
            
            visited.add((x, y))
            count += 1
            
            # Add neighbors (with torus wrapping)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = (x + dx) % width
                ny = (y + dy) % height
                if (nx, ny) not in visited:
                    queue.append((nx, ny))
        
        return count
    
    def _emergency_a_star(self, game, start, goal):
        """A* pathfinding on torus board"""
        width, height = len(game.board.grid[0]), len(game.board.grid)
        
        def torus_manhattan(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            dx = min(abs(x2 - x1), width - abs(x2 - x1))
            dy = min(abs(y2 - y1), height - abs(y2 - y1))
            return dx + dy
        
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current == goal:
                break
            
            x, y = current
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = (x + dx) % width
                ny = (y + dy) % height
                next_pos = (nx, ny)
                
                # Skip occupied cells
                if game.board.grid[ny][nx] != 0:
                    continue
                
                new_cost = cost_so_far[current] + 1
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + torus_manhattan(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # Reconstruct path
        if goal not in came_from:
            return None
        
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
    
    def smart_act(self, state, valid_actions, game, my_agent, opponent_agent):
        """
        Smart action selection with emergency override
        If in danger, use emergency pathfinding. Otherwise use normal DQN.
        """
        # Check if we're in immediate danger
        if self.is_in_danger(game, my_agent, opponent_agent, threshold=1):
            # EMERGENCY MODE: Use A* pathfinding to escape
            emergency_direction = self.find_emergency_escape(game, my_agent, opponent_agent)
            
            if emergency_direction:
                # Convert direction to action index
                direction_to_idx = {
                    Direction.RIGHT: 0,
                    Direction.LEFT: 1,
                    Direction.DOWN: 2,
                    Direction.UP: 3
                }
                direction_idx = direction_to_idx.get(emergency_direction, 0)
                action = direction_idx * 2  # No boost in emergency
                
                # Store for tracking
                self.last_emergency_action = action
                
                if action in valid_actions:
                    return action
        
        # Normal mode: use existing select_action logic
        # Convert valid_actions back to game context
        # This is a simplified version - in practice, integrate with existing logic
        self.last_emergency_action = None
        
        # Use existing epsilon-greedy selection
        if random.random() < self.epsilon:
            return random.choice(valid_actions) if valid_actions else 0
        else:
            # Use DQN to select best action
            # For now, just pick first valid action (you can enhance this)
            return valid_actions[0] if valid_actions else 0
    
    # ==================== END EMERGENCY PATHFINDING ====================
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Extract components (simplified - needs proper implementation)
        # This is a placeholder for the full training loop
        
        losses = []
        for exp in batch:
            # Compute loss and update (simplified)
            pass
        
        # Decay epsilon
        if self.training_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return np.mean(losses) if losses else None
    
    def save_model(self, path="rl_agent_model.pth"):
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="rl_agent_model.pth"):
        """Load model weights"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            self.episodes = checkpoint.get('episodes', 0)
            print(f"Model loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"No model found at {path}, starting fresh")
            return False


# Global RL agent instance
_rl_agent = None

def get_rl_agent():
    """Get or create global RL agent instance"""
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = HeuristicGuidedDQNAgent()
        # Try to load existing model
        _rl_agent.load_model()
    return _rl_agent


def send_move_rl_agent(game, player_number=2):
    """
    RL agent move function
    Compatible with local_judge.py interface
    """
    agent = get_rl_agent()
    direction, use_boost = agent.select_action(game, player_number, training=False)
    return direction, use_boost


def send_move_agent2(game):
    """Agent 2 using RL (for local_judge compatibility)"""
    return send_move_rl_agent(game, player_number=2)
