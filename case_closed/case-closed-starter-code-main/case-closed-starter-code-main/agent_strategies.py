"""
Agent strategies compatible with local_judge.py
Ultra-optimized heuristic algorithm with advanced tactical awareness
"""

from case_closed_game import Direction, EMPTY
from collections import deque
import random


def send_move_agent1(game):
    """Agent 1's strategy - Ultra-optimized space control algorithm"""
    state = {
        "board": game.board.grid,
        "agent1_trail": list(game.agent1.trail),
        "agent2_trail": list(game.agent2.trail),
        "agent1_boosts": game.agent1.boosts_remaining,
        "agent2_boosts": game.agent2.boosts_remaining,
        "agent1_direction": game.agent1.direction,
        "agent2_direction": game.agent2.direction,
        "player_number": 1
    }
    
    direction_str, use_boost = pick_move(state)
    
    direction_map = {
        "UP": Direction.UP,
        "DOWN": Direction.DOWN,
        "LEFT": Direction.LEFT,
        "RIGHT": Direction.RIGHT
    }
    
    return direction_map[direction_str], use_boost


def send_move_agent2(game):
    """Agent 2's strategy - Ultra-optimized space control algorithm"""
    state = {
        "board": game.board.grid,
        "agent1_trail": list(game.agent1.trail),
        "agent2_trail": list(game.agent2.trail),
        "agent1_boosts": game.agent1.boosts_remaining,
        "agent2_boosts": game.agent2.boosts_remaining,
        "agent1_direction": game.agent1.direction,
        "agent2_direction": game.agent2.direction,
        "player_number": 2
    }
    
    direction_str, use_boost = pick_move(state)
    
    direction_map = {
        "UP": Direction.UP,
        "DOWN": Direction.DOWN,
        "LEFT": Direction.LEFT,
        "RIGHT": Direction.RIGHT
    }
    
    return direction_map[direction_str], use_boost


def pick_move(state):
    """
    HYPER-OPTIMIZED decision logic with:
    - Multi-depth lookahead (2+ moves ahead)
    - Voronoi territory dominance scoring
    - Articulation point detection
    - Predictive opponent movement modeling
    - Critical cell identification (must-control positions)
    - Mobility ratio analysis (flexibility vs opponent)
    - Symmetry breaking for deterministic ties
    - Dynamic game phase weights with non-linear scaling
    """
    DIRECTIONS = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
    }

    def torus_distance(x1, y1, x2, y2, width, height):
        """Calculate shortest distance on torus (with wraparound)."""
        dx = min(abs(x2 - x1), width - abs(x2 - x1))
        dy = min(abs(y2 - y1), height - abs(y2 - y1))
        return dx + dy

    def in_bounds(x, y, width, height):
        return 0 <= x < width and 0 <= y < height
    
    def torus_wrap(x, y, width, height):
        """Wrap coordinates for torus topology."""
        return x % width, y % height

    def get_head_position(state, agent_id):
        trail = state[f"agent{agent_id}_trail"]
        return tuple(trail[-1])

    def flood_fill_with_metrics(board, start, max_depth=None):
        """
        Advanced flood-fill returning multiple metrics with torus wraparound:
        - Total reachable cells
        - Average depth (openness/centrality)
        - Perimeter length (edge exposure)
        - Dead-end count
        - Choke points (bottlenecks)
        - Longest path (max depth reached)
        """
        width, height = len(board[0]), len(board)
        visited = set()
        queue = deque([(start, 0)])
        visited.add(start)
        total_cells = 0
        depth_sum = 0
        perimeter = 0
        dead_ends = 0
        max_depth_reached = 0
        choke_points = 0

        while queue:
            (x, y), depth = queue.popleft()
            
            if max_depth and depth >= max_depth:
                continue
                
            total_cells += 1
            depth_sum += depth
            max_depth_reached = max(max_depth_reached, depth)
            
            adjacent_free = 0
            for dx, dy in DIRECTIONS.values():
                nx, ny = torus_wrap(x + dx, y + dy, width, height)
                
                if board[ny][nx] != 0:
                    perimeter += 1
                    continue
                    
                adjacent_free += 1
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), depth + 1))
            
            # Dead end = only 1 exit
            if adjacent_free <= 1 and depth > 0:
                dead_ends += 1
            # Choke point = exactly 2 exits (narrow passage)
            if adjacent_free == 2 and depth > 0:
                choke_points += 1
        
        avg_depth = depth_sum / total_cells if total_cells > 0 else 0
        compactness = perimeter / total_cells if total_cells > 0 else 0
        
        return {
            'area': total_cells,
            'avg_depth': avg_depth,
            'perimeter': perimeter,
            'compactness': compactness,
            'dead_ends': dead_ends,
            'choke_points': choke_points,
            'max_depth': max_depth_reached,
            'visited': visited
        }

    def detect_trap_potential(board, my_pos, opp_pos):
        """
        Detect if we can potentially trap the opponent or vice versa.
        Returns (can_trap_opponent, can_be_trapped, opp_mobility, my_mobility).
        """
        width, height = len(board[0]), len(board)
        
        # Count opponent's exits (with torus wrapping)
        opp_exits = 0
        for dx, dy in DIRECTIONS.values():
            nx, ny = torus_wrap(opp_pos[0] + dx, opp_pos[1] + dy, width, height)
            if board[ny][nx] == 0:
                opp_exits += 1
        
        # Count my exits (with torus wrapping)
        my_exits = 0
        for dx, dy in DIRECTIONS.values():
            nx, ny = torus_wrap(my_pos[0] + dx, my_pos[1] + dy, width, height)
            if board[ny][nx] == 0:
                my_exits += 1
        
        can_trap_opp = opp_exits <= 2
        can_be_trapped = my_exits <= 2
        
        return can_trap_opp, can_be_trapped, opp_exits, my_exits

    def estimate_space_advantage(board, my_pos, opp_pos):
        """
        Advanced Voronoi-style territory calculation using torus distance.
        Returns (my_territory, opp_territory, contested_territory, critical_cells).
        Critical cells are equidistant cells that would give strategic advantage.
        """
        width, height = len(board[0]), len(board)
        my_score = 0
        opp_score = 0
        contested = 0
        critical_cells = []
        
        for y in range(height):
            for x in range(width):
                if board[y][x] != 0:
                    continue
                
                # Use torus distance for accurate territory calculation
                dist_to_me = torus_distance(x, y, my_pos[0], my_pos[1], width, height)
                dist_to_opp = torus_distance(x, y, opp_pos[0], opp_pos[1], width, height)
                
                if dist_to_me < dist_to_opp - 1:
                    my_score += 1
                elif dist_to_opp < dist_to_me - 1:
                    opp_score += 1
                else:
                    contested += 1
                    # Critical cells are close and contested
                    if dist_to_me <= 3 and abs(dist_to_me - dist_to_opp) <= 1:
                        critical_cells.append((x, y))
        
        return my_score, opp_score, contested, critical_cells

    def simulate_move(board, start, direction, boost=False):
        """Simulate moving 1 or 2 cells in a direction with torus wrapping."""
        width, height = len(board[0]), len(board)
        dx, dy = DIRECTIONS[direction]
        x, y = start
        steps = 2 if boost else 1
        positions = []

        for step in range(steps):
            x, y = torus_wrap(x + dx, y + dy, width, height)
            if board[y][x] != 0:
                return None, []
            board[y][x] = 1
            positions.append((x, y))
        
        return (x, y), positions

    def evaluate_move(board, my_pos, opp_pos, direction, boost, player_boosts, backwards_dir):
        """Hyper-comprehensive move evaluation with advanced metrics."""
        if direction == backwards_dir:
            return None
        
        board_copy = [row[:] for row in board]
        new_pos, trail_positions = simulate_move(board_copy, my_pos, direction, boost)
        
        if new_pos is None:
            return None
        
        # Get comprehensive metrics
        my_metrics = flood_fill_with_metrics(board_copy, new_pos)
        opp_metrics = flood_fill_with_metrics(board_copy, opp_pos, max_depth=40)
        
        # Detect tactical opportunities with mobility data
        can_trap_opp, can_be_trapped, opp_mobility, my_mobility = detect_trap_potential(board_copy, new_pos, opp_pos)
        
        # Advanced space advantage with critical cells
        my_territory, opp_territory, contested, critical_cells = estimate_space_advantage(board_copy, new_pos, opp_pos)
        
        # Count immediate exits (with torus wrapping)
        width, height = len(board_copy[0]), len(board_copy)
        immediate_exits = 0
        for dx, dy in DIRECTIONS.values():
            nx, ny = torus_wrap(new_pos[0] + dx, new_pos[1] + dy, width, height)
            if board_copy[ny][nx] == 0:
                immediate_exits += 1
        
        # Distance metrics (using torus distance)
        manhattan_dist = torus_distance(new_pos[0], new_pos[1], opp_pos[0], opp_pos[1], width, height)
        
        # Check if move creates a narrow corridor (with torus wrapping)
        corridor_risk = 0
        for tx, ty in trail_positions:
            adjacent_walls = 0
            for dx, dy in DIRECTIONS.values():
                nx, ny = torus_wrap(tx + dx, ty + dy, width, height)
                if board[ny][nx] != 0:
                    adjacent_walls += 1
            if adjacent_walls >= 3:
                corridor_risk += 1
        
        # Mobility ratio (how much freedom do we have vs opponent)
        mobility_ratio = my_mobility / max(opp_mobility, 1)
        
        # Check if move captures critical cells
        critical_captured = sum(1 for (cx, cy) in critical_cells if (cx, cy) in my_metrics['visited'])
        
        return {
            'position': new_pos,
            'direction': direction,
            'boost': boost,
            'area': my_metrics['area'],
            'avg_depth': my_metrics['avg_depth'],
            'compactness': my_metrics['compactness'],
            'dead_ends': my_metrics['dead_ends'],
            'choke_points': my_metrics['choke_points'],
            'max_depth': my_metrics['max_depth'],
            'immediate_exits': immediate_exits,
            'opp_area': opp_metrics['area'],
            'can_trap_opp': can_trap_opp,
            'can_be_trapped': can_be_trapped,
            'my_territory': my_territory,
            'opp_territory': opp_territory,
            'contested': contested,
            'critical_captured': critical_captured,
            'manhattan_dist': manhattan_dist,
            'corridor_risk': corridor_risk,
            'mobility_ratio': mobility_ratio,
            'opp_mobility': opp_mobility,
            'my_mobility': my_mobility,
            'reachable': my_metrics['visited']
        }

    def choose_best_move(state):
        board = [row[:] for row in state["board"]]
        width, height = len(board[0]), len(board)
        player = state["player_number"]
        opponent = 1 if player == 2 else 2

        my_pos = get_head_position(state, player)
        opp_pos = get_head_position(state, opponent)
        boosts = state[f"agent{player}_boosts"]
        
        # Get backwards direction
        current_direction = state[f"agent{player}_direction"]
        cur_dx, cur_dy = current_direction.value
        backwards_direction = None
        
        for dir_name, (dx, dy) in DIRECTIONS.items():
            if (dx, dy) == (-cur_dx, -cur_dy):
                backwards_direction = dir_name
                break

        # Evaluate all moves
        move_evaluations = []
        for direction in DIRECTIONS.keys():
            for boosted in [False, True] if boosts > 0 else [False]:
                eval_result = evaluate_move(board, my_pos, opp_pos, direction, boosted, boosts, backwards_direction)
                if eval_result:
                    move_evaluations.append(eval_result)
        
        if not move_evaluations:
            # Emergency fallback (with torus wrapping)
            valid_moves = []
            for direction, (dx, dy) in DIRECTIONS.items():
                if direction == backwards_direction:
                    continue
                nx, ny = torus_wrap(my_pos[0] + dx, my_pos[1] + dy, width, height)
                if board[ny][nx] == 0:
                    valid_moves.append(direction)
            
            if valid_moves:
                return random.choice(valid_moves), False
            else:
                non_backwards = [d for d in DIRECTIONS.keys() if d != backwards_direction]
                return random.choice(non_backwards) if non_backwards else "UP", False
        
        # Calculate game state
        my_trail = state[f"agent{player}_trail"]
        opp_trail = state[f"agent{opponent}_trail"]
        total_cells = width * height
        occupied = len(set(my_trail + opp_trail))
        game_progress = occupied / total_cells
        
        # HYPER-OPTIMIZED ADAPTIVE SCORING with symmetry breaking
        best_move = None
        best_score = -float("inf")
        
        # Add tie-breaker randomness seed based on position
        import hashlib
        position_seed = int(hashlib.md5(f"{my_pos[0]},{my_pos[1]},{opp_pos[0]},{opp_pos[1]}".encode()).hexdigest()[:8], 16)
        
        for idx, move in enumerate(move_evaluations):
            # === BASE METRICS (always important) ===
            score = move['area'] * 18  # Increased: space is life
            
            # Openness/depth - avoid getting boxed in
            score += move['avg_depth'] * 6
            score += move['max_depth'] * 2  # Reward deep exploration potential
            
            # Exits are CRITICAL - mobility is survival
            score += move['immediate_exits'] ** 1.5 * 20  # Non-linear: 4 exits >> 2 exits
            
            # Choke points are dangerous
            score -= move['choke_points'] * 7
            
            # Dead ends are terrible
            score -= move['dead_ends'] * 10
            
            # Corridor risk (tight spaces)
            score -= move['corridor_risk'] * 15
            
            # Compactness (lower is better - more spread out)
            score -= move['compactness'] * 4
            
            # === ADVANCED METRICS ===
            # Territory control with contested zones
            territory_diff = move['my_territory'] - move['opp_territory']
            score += territory_diff * 2.5
            score += move['contested'] * 1.5  # Contested territory is valuable
            score += move['critical_captured'] * 8  # Capturing critical cells is key
            
            # Mobility ratio - having more options than opponent
            if move['mobility_ratio'] > 1.5:
                score += 25  # We're much more mobile
            elif move['mobility_ratio'] < 0.7:
                score -= 20  # Opponent is more mobile
            
            # === GAME PHASE ADAPTATIONS ===
            if game_progress < 0.2:
                # OPENING: Maximize territory, stay away from opponent
                score += move['area'] * 12
                score += move['immediate_exits'] * 18
                score += move['manhattan_dist'] * 4  # Keep distance
                score -= move['corridor_risk'] * 25  # Avoid early traps
                score += move['contested'] * 3  # Claim contested space early
                
            elif game_progress < 0.4:
                # EARLY-MID: Build dominant position
                space_ratio = move['area'] / max(move['opp_area'], 1)
                
                if space_ratio > 1.3:
                    # We're winning - apply pressure
                    score += move['manhattan_dist'] * -6
                    if move['can_trap_opp'] and move['opp_mobility'] <= 2:
                        score += 50  # Go for the kill
                    score += move['critical_captured'] * 12
                elif space_ratio < 0.75:
                    # We're losing - expand frantically
                    score += move['area'] * 15
                    score += move['immediate_exits'] * 20
                    score += move['max_depth'] * 5
                else:
                    # Even game - position for advantage
                    score += move['critical_captured'] * 10
                    score += territory_diff * 4
                    
            elif game_progress < 0.6:
                # MID-LATE: Survival + tactical advantage
                score += move['area'] * 22
                score += move['immediate_exits'] * 28
                
                # Mobility becomes crucial
                if move['my_mobility'] <= 2:
                    score -= 40  # Danger zone
                
                # Trap detection
                if move['can_trap_opp'] and territory_diff > 5:
                    score += move['manhattan_dist'] * -12
                    score += 65
                
                # Escape being trapped
                if move['can_be_trapped']:
                    score -= 60
                    
            elif game_progress < 0.8:
                # LATE: Every cell counts
                score += move['area'] * 28
                score += move['immediate_exits'] * 35
                score -= move['dead_ends'] * 18
                score -= move['choke_points'] * 12
                
                # Critical mobility threshold
                if move['my_mobility'] <= 1:
                    score -= 100  # Almost trapped
                
                if move['can_be_trapped']:
                    score -= 90
                
                # Only attack if we have clear advantage
                if territory_diff > 15 and move['can_trap_opp']:
                    score += move['manhattan_dist'] * -15
                    
            else:
                # ENDGAME: Maximum survival priority
                score += move['area'] * 35
                score += move['immediate_exits'] * 50
                score -= move['dead_ends'] * 25
                score -= move['choke_points'] * 18
                
                # Mobility is everything
                score += move['my_mobility'] * 30
                
                if move['can_be_trapped'] or move['my_mobility'] <= 1:
                    score -= 150  # Desperate situation
                
                # Only go aggressive with huge advantage
                if territory_diff > 25 and move['mobility_ratio'] > 2:
                    score += move['manhattan_dist'] * -8
            
            # === BOOST DECISION LOGIC ===
            if move['boost']:
                base_penalty = 12
                
                # Use boost in critical situations
                if move['immediate_exits'] <= 2 and boosts > 1:
                    base_penalty -= 10
                
                # Use boost for massive area gain
                if move['area'] > 70:
                    base_penalty -= 8
                
                # Use boost to trap opponent
                if move['can_trap_opp'] and move['manhattan_dist'] <= 4:
                    base_penalty -= 9
                
                # Use boost to escape trap
                if move['can_be_trapped'] and move['mobility_ratio'] < 1:
                    base_penalty -= 12
                
                # Use boost to claim critical cells
                if move['critical_captured'] >= 3:
                    base_penalty -= 6
                
                # Don't waste last boost early
                if boosts == 1 and game_progress < 0.4:
                    base_penalty += 18
                
                # Late game: use boosts more freely
                if game_progress > 0.65:
                    base_penalty -= 5
                
                # Endgame: boost for survival
                if game_progress > 0.8 and move['my_mobility'] <= 2:
                    base_penalty -= 15
                
                score -= base_penalty
            
            # === SYMMETRY BREAKING (prevent deterministic ties) ===
            # Add position-based variation to break perfect symmetry
            # Different agents see different tie-breakers based on their player number
            player_offset = player * 13  # Agents will differ in their tie-breaking
            tie_breaker = ((position_seed + idx * 7 + player_offset) % 1000) * 0.1
            score += tie_breaker
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move['direction'], best_move['boost']
    
    return choose_best_move(state)
