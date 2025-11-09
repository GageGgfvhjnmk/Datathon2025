"""
tronbot_algorithms.py

Complete implementations of classic TronBot algorithms with COMPATIBLE interface.
Now returns (Direction, bool) instead of (direction_idx, use_boost)
"""
import random
import math
from collections import deque
from heapq import heappush, heappop
from case_closed_game import Direction, EMPTY


class TronBotAlgorithms:
    """
    Collection of classic TronBot algorithms with COMPATIBLE interface.
    Each method returns (Direction, use_boost) for the given game state.
    """
    
    @staticmethod
    def flood_fill_bot(game_state, my_agent, opponent, depth=3):
        """
        Classic FloodFill algorithm - maximizes reachable space.
        Returns (Direction, use_boost) for compatibility.
        """
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
        
        def flood_fill_area(start_pos, simulated_board=None):
            """Calculate reachable area from position using BFS"""
            if simulated_board is None:
                simulated_board = {}
                for y in range(game_state.board.height):
                    for x in range(game_state.board.width):
                        simulated_board[(x, y)] = game_state.board.get_cell_state((x, y))
            
            visited = set()
            queue = deque([start_pos])
            area_size = 0
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                    
                visited.add(current)
                area_size += 1
                
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    next_pos = (
                        (current[0] + dx) % game_state.board.width,
                        (current[1] + dy) % game_state.board.height
                    )
                    
                    if (next_pos not in visited and 
                        simulated_board.get(next_pos, 0) == EMPTY):
                        queue.append(next_pos)
            
            return area_size
        
        # Evaluate each possible move
        best_direction = None
        best_score = -float('inf')
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for direction in directions:
            if direction == forbidden_direction:
                continue
                
            dx, dy = direction.value
            next_pos = (
                (my_pos[0] + dx) % game_state.board.width,
                (my_pos[1] + dy) % game_state.board.height
            )
            
            # Check if move is valid
            if (game_state.board.get_cell_state(next_pos) == EMPTY or 
                next_pos == opp_pos):
                
                # Calculate reachable area from this position
                area_size = flood_fill_area(next_pos)
                
                # Consider opponent's move as well
                opp_area = 0
                if depth > 1:
                    # Simulate opponent's best response
                    opp_best_area = 0
                    for opp_dx, opp_dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        opp_next_pos = (
                            (opp_pos[0] + opp_dx) % game_state.board.width,
                            (opp_pos[1] + opp_dy) % game_state.board.height
                        )
                        if (game_state.board.get_cell_state(opp_next_pos) == EMPTY or 
                            opp_next_pos == my_pos):
                            opp_area_temp = flood_fill_area(opp_next_pos)
                            opp_best_area = max(opp_best_area, opp_area_temp)
                    opp_area = opp_best_area
                
                # Score: our area minus opponent's area
                score = area_size - (opp_area * 0.5)
                
                # Bonus for staying alive (having multiple exits)
                safe_exits = 0
                for test_dx, test_dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    test_pos = (
                        (next_pos[0] + test_dx) % game_state.board.width,
                        (next_pos[1] + test_dy) % game_state.board.height
                    )
                    if game_state.board.get_cell_state(test_pos) == EMPTY:
                        safe_exits += 1
                
                score += safe_exits * 2
                
                if score > best_score:
                    best_score = score
                    best_direction = direction
        
        if best_direction is not None:
            # Use boost strategically - when we have good space advantage
            use_boost = (my_agent.boosts_remaining > 0 and 
                        best_score > 50 and  # Good position
                        TronBotAlgorithms._torus_manhattan(my_pos, opp_pos, game_state.board) < 6)
            return best_direction, use_boost
        
        # Fallback: any valid move
        for direction in directions:
            if direction != forbidden_direction:
                return direction, False
        
        return current_direction, False
    
    @staticmethod
    def wall_hugger_bot(game_state, my_agent, opponent, aggression=0.3):
        """
        Wall Hugger algorithm - prefers to stay near walls and corners.
        Returns (Direction, use_boost) for compatibility.
        """
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
        
        def distance_to_walls(pos, board):
            """Calculate distance to nearest walls"""
            left_dist = pos[0]
            right_dist = board.width - 1 - pos[0]
            top_dist = pos[1]
            bottom_dist = board.height - 1 - pos[1]
            return min(left_dist, right_dist, top_dist, bottom_dist)
        
        # Evaluate each direction
        best_direction = None
        best_score = -float('inf')
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for direction in directions:
            if direction == forbidden_direction:
                continue
                
            dx, dy = direction.value
            next_pos = (
                (my_pos[0] + dx) % game_state.board.width,
                (my_pos[1] + dy) % game_state.board.height
            )
            
            # Check if move is valid
            if (game_state.board.get_cell_state(next_pos) == EMPTY or 
                next_pos == opp_pos):
                
                score = 0
                
                # Wall hugging component (defensive)
                wall_dist = distance_to_walls(next_pos, game_state.board)
                wall_score = (10 - wall_dist) * 2  # Prefer being closer to walls
                
                # Opponent distance component (aggressive)
                opp_dist = TronBotAlgorithms._torus_manhattan(next_pos, opp_pos, game_state.board)
                opp_score = (10 - opp_dist) * 3  # Prefer being closer to opponent
                
                # Combine based on aggression parameter
                score = (wall_score * (1 - aggression)) + (opp_score * aggression)
                
                # Safety bonus - check future mobility
                safe_moves = 0
                for test_dx, test_dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    test_pos = (
                        (next_pos[0] + test_dx) % game_state.board.width,
                        (next_pos[1] + test_dy) % game_state.board.height
                    )
                    if game_state.board.get_cell_state(test_pos) == EMPTY:
                        safe_moves += 1
                
                score += safe_moves * 5
                
                # Corner bonus
                if wall_dist <= 2:
                    score += 10
                
                if score > best_score:
                    best_score = score
                    best_direction = direction
        
        if best_direction is not None:
            # Use boost when we're in a good position to attack
            use_boost = (my_agent.boosts_remaining > 0 and 
                        aggression > 0.5 and  # Aggressive mode
                        TronBotAlgorithms._torus_manhattan(my_pos, opp_pos, game_state.board) < 4)
            return best_direction, use_boost
        
        # Fallback
        for direction in directions:
            if direction != forbidden_direction:
                return direction, False
        
        return current_direction, False
    
    @staticmethod
    def minimax_bot(game_state, my_agent, opponent, depth=3):
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (Direction, use_boost) for compatibility.
        """
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        def evaluate_position(pos, is_maximizing, board):
            """Evaluate how good a position is for the given player"""
            score = 0
            
            # Mobility - number of safe moves
            safe_moves = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                check_pos = (
                    (pos[0] + dx) % board.width,
                    (pos[1] + dy) % board.height
                )
                if board.get_cell_state(check_pos) == EMPTY:
                    safe_moves += 1
            
            score += safe_moves * 20
            
            # Territory control - flood fill area
            territory = TronBotAlgorithms._flood_fill_single(pos, game_state)
            score += territory * 0.1
            
            # Opponent proximity
            dist = TronBotAlgorithms._torus_manhattan(pos, opp_pos if is_maximizing else my_pos, board)
            if is_maximizing:
                score -= dist * 2  # Prefer being closer to opponent when maximizing
            else:
                score += dist * 2  # Prefer being farther from opponent when minimizing
            
            # Wall proximity (slight preference for center)
            wall_dist = min(pos[0], board.width-1-pos[0], pos[1], board.height-1-pos[1])
            score += (10 - wall_dist) * 0.5
            
            return score
        
        def minimax(my_pos, opp_pos, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
            if depth == 0:
                return evaluate_position(my_pos if is_maximizing else opp_pos, is_maximizing, game_state.board)
            
            if is_maximizing:
                max_eval = -float('inf')
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_pos = (
                        (my_pos[0] + dx) % game_state.board.width,
                        (my_pos[1] + dy) % game_state.board.height
                    )
                    if game_state.board.get_cell_state(new_pos) == EMPTY:
                        eval = minimax(new_pos, opp_pos, depth-1, False, alpha, beta)
                        max_eval = max(max_eval, eval)
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
                return max_eval
            else:
                min_eval = float('inf')
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_pos = (
                        (opp_pos[0] + dx) % game_state.board.width,
                        (opp_pos[1] + dy) % game_state.board.height
                    )
                    if game_state.board.get_cell_state(new_pos) == EMPTY:
                        eval = minimax(my_pos, new_pos, depth-1, True, alpha, beta)
                        min_eval = min(min_eval, eval)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
                return min_eval
        
        # Find best move for current player
        current_direction = my_agent.direction
        reverse_directions = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        forbidden_direction = reverse_directions.get(current_direction)
        
        best_score = -float('inf')
        best_direction = None
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for direction in directions:
            if direction == forbidden_direction:
                continue
                
            dx, dy = direction.value
            new_pos = (
                (my_pos[0] + dx) % game_state.board.width,
                (my_pos[1] + dy) % game_state.board.height
            )
            
            if (game_state.board.get_cell_state(new_pos) == EMPTY or 
                new_pos == opp_pos):
                
                score = minimax(new_pos, opp_pos, depth-1, False)
                if score > best_score:
                    best_score = score
                    best_direction = direction
        
        if best_direction is not None:
            # Use boost when we have a clear advantage
            use_boost = (my_agent.boosts_remaining > 0 and 
                        best_score > 100 and  # Strong position
                        depth >= 2)  # We looked ahead enough to be confident
            return best_direction, use_boost
        
        # Fallback
        for direction in directions:
            if direction != forbidden_direction:
                return direction, False
        
        return current_direction, False
    
    @staticmethod
    def space_invader_bot(game_state, my_agent, opponent):
        """
        Space Invader algorithm - focuses on claiming the largest empty regions.
        Returns (Direction, use_boost) for compatibility.
        """
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        def find_empty_regions():
            """Find all connected empty regions and their sizes"""
            visited = set()
            regions = []
            
            for x in range(game_state.board.width):
                for y in range(game_state.board.height):
                    pos = (x, y)
                    if pos not in visited and game_state.board.get_cell_state(pos) == EMPTY:
                        # New region found
                        region = set()
                        queue = deque([pos])
                        
                        while queue:
                            current = queue.popleft()
                            if current in visited:
                                continue
                                
                            visited.add(current)
                            region.add(current)
                            
                            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                                next_pos = (
                                    (current[0] + dx) % game_state.board.width,
                                    (current[1] + dy) % game_state.board.height
                                )
                                if (next_pos not in visited and 
                                    game_state.board.get_cell_state(next_pos) == EMPTY):
                                    queue.append(next_pos)
                        
                        regions.append(region)
            
            return regions
        
        regions = find_empty_regions()
        if not regions:
            # No empty space - fallback to wall hugger
            return TronBotAlgorithms.wall_hugger_bot(game_state, my_agent, opponent)
        
        # Find largest region
        largest_region = max(regions, key=len)
        
        # Check if we should try to cut off opponent
        opponent_region = None
        for region in regions:
            if opp_pos in region:
                opponent_region = region
                break
        
        # If opponent is in a small region, try to trap them
        if opponent_region and len(opponent_region) <= 10:
            target_region = opponent_region
            strategy = "TRAPPING"
        else:
            target_region = largest_region
            strategy = "EXPANDING"
        
        # Find closest point in target region
        closest_point = None
        min_distance = float('inf')
        
        for point in target_region:
            dist = TronBotAlgorithms._torus_manhattan(my_pos, point, game_state.board)
            if dist < min_distance:
                min_distance = dist
                closest_point = point
        
        if closest_point is None:
            return TronBotAlgorithms.wall_hugger_bot(game_state, my_agent, opponent)
        
        # Move toward closest point in target region
        current_direction = my_agent.direction
        reverse_directions = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP
        }
        forbidden_direction = reverse_directions.get(current_direction)
        
        best_direction = None
        min_distance_to_target = float('inf')
        directions = [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP]
        
        for direction in directions:
            if direction == forbidden_direction:
                continue
                
            dx, dy = direction.value
            next_pos = (
                (my_pos[0] + dx) % game_state.board.width,
                (my_pos[1] + dy) % game_state.board.height
            )
            
            if (game_state.board.get_cell_state(next_pos) == EMPTY or 
                next_pos == opp_pos):
                
                dist = TronBotAlgorithms._torus_manhattan(next_pos, closest_point, game_state.board)
                if dist < min_distance_to_target:
                    min_distance_to_target = dist
                    best_direction = direction
        
        if best_direction is not None:
            # Use boost when we're expanding into large territory
            use_boost = (my_agent.boosts_remaining > 0 and 
                        strategy == "EXPANDING" and 
                        len(largest_region) > 50)  # Large territory to claim
            return best_direction, use_boost
        
        # Fallback
        for direction in directions:
            if direction != forbidden_direction:
                return direction, False
        
        return current_direction, False
    
    @staticmethod
    def hybrid_bot(game_state, my_agent, opponent, phase_detection=True):
        """
        Hybrid strategy that switches between algorithms based on game phase.
        Returns (Direction, use_boost) for compatibility.
        """
        my_pos = my_agent.trail[-1]
        opp_pos = opponent.trail[-1]
        
        # Detect game phase based on board occupancy
        empty_cells = 0
        for x in range(game_state.board.width):
            for y in range(game_state.board.height):
                if game_state.board.get_cell_state((x, y)) == EMPTY:
                    empty_cells += 1
        
        total_cells = game_state.board.width * game_state.board.height
        occupancy_ratio = 1 - (empty_cells / total_cells)
        
        # Phase detection
        if occupancy_ratio < 0.3:
            # Early game - focus on space control
            return TronBotAlgorithms.space_invader_bot(game_state, my_agent, opponent)
        elif occupancy_ratio < 0.7:
            # Mid game - balanced approach
            return TronBotAlgorithms.flood_fill_bot(game_state, my_agent, opponent, depth=2)
        else:
            # Late game - aggressive trapping
            return TronBotAlgorithms.minimax_bot(game_state, my_agent, opponent, depth=3)
    
    # Helper methods
    @staticmethod
    def _flood_fill_single(start_pos, game_state):
        """Quick flood fill for a single position"""
        visited = set()
        queue = deque([start_pos])
        area_size = 0
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
                
            visited.add(current)
            area_size += 1
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_pos = (
                    (current[0] + dx) % game_state.board.width,
                    (current[1] + dy) % game_state.board.height
                )
                
                if (next_pos not in visited and 
                    game_state.board.get_cell_state(next_pos) == EMPTY):
                    queue.append(next_pos)
        
        return area_size
    
    @staticmethod
    def _torus_manhattan(pos1, pos2, board):
        """Calculate Manhattan distance on torus board"""
        dx = min(abs(pos1[0] - pos2[0]), board.width - abs(pos1[0] - pos2[0]))
        dy = min(abs(pos1[1] - pos2[1]), board.height - abs(pos1[1] - pos2[1]))
        return dx + dy


# Convenience functions for easy access
def get_tronbot_opponents():
    """Return all TronBot algorithms as a list of (name, function) pairs"""
    return [
        ("FloodFill", lambda g, m, o: TronBotAlgorithms.flood_fill_bot(g, m, o)),
        ("WallHugger", lambda g, m, o: TronBotAlgorithms.wall_hugger_bot(g, m, o)),
        ("Minimax", lambda g, m, o: TronBotAlgorithms.minimax_bot(g, m, o, depth=3)),
        ("SpaceInvader", TronBotAlgorithms.space_invader_bot),
        ("Hybrid", TronBotAlgorithms.hybrid_bot),
    ]