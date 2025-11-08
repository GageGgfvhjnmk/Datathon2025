"""
Rammer Agent Strategy - Compatible with local_judge.py
This can be imported and called from local_judge.py

This implements the "rammer" strategy with:
- Backwards movement prevention
- Safety-first approach (checks all moves before sorting)
- Torus wrapping support
- Distance-based move prioritization
- Smart boost usage
"""

from case_closed_game import Direction, EMPTY

def send_move_agent1(game):
    """
    Agent 1's strategy - Rammer with safety and backwards prevention
    
    Args:
        game: Game object with .agent1, .agent2, .board attributes
    
    Returns:
        tuple: (Direction, bool) - direction to move and whether to use boost
    """
    my_agent = game.agent1
    opponent = game.agent2
    my_pos = my_agent.trail[-1]
    opponent_pos = opponent.trail[-1]
    board = game.board
    
    # Calculate distance considering torus wrapping
    def torus_distance(pos1, pos2, board):
        x1, y1 = pos1
        x2, y2 = pos2
        
        # Calculate directional distances with torus wrapping
        # X-axis distances
        dx_right = (x2 - x1) % board.width  # Distance going right (wraps around)
        dx_left = (x1 - x2) % board.width   # Distance going left (wraps around)
        
        # Y-axis distances
        dy_down = (y2 - y1) % board.height  # Distance going down (wraps around)
        dy_up = (y1 - y2) % board.height    # Distance going up (wraps around)
        
        # Minimum distances for each axis
        dx = min(dx_left, dx_right)
        dy = min(dy_up, dy_down)
        
        return dx + dy
    
    # Determine best direction to move toward opponent
    def get_direction_to_target(my_pos, target_pos, board, my_trail):
        x1, y1 = my_pos
        x2, y2 = target_pos
        
        # Determine current direction (to avoid moving backwards)
        current_direction = None
        if len(my_trail) >= 2:
            prev_pos = my_trail[-2]
            dx = (x1 - prev_pos[0]) % board.width
            dy = (y1 - prev_pos[1]) % board.height
            
            # Normalize for torus wrapping
            if dx == board.width - 1:
                dx = -1
            if dy == board.height - 1:
                dy = -1
            
            if dx == 1 and dy == 0:
                current_direction = "RIGHT"
            elif dx == -1 and dy == 0:
                current_direction = "LEFT"
            elif dx == 0 and dy == 1:
                current_direction = "DOWN"
            elif dx == 0 and dy == -1:
                current_direction = "UP"
        
        # Define opposite directions
        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        backwards = opposite.get(current_direction) if current_direction else None
        
        # Calculate directional distances with torus wrapping
        dx_right = (x2 - x1) % board.width
        dx_left = (x1 - x2) % board.width
        dy_down = (y2 - y1) % board.height
        dy_up = (y1 - y2) % board.height
        
        # Create a list of (direction, distance, next_pos) tuples
        direction_data = [
            ("RIGHT", dx_right, ((x1 + 1) % board.width, y1)),
            ("LEFT", dx_left, ((x1 - 1) % board.width, y1)),
            ("DOWN", dy_down, (x1, (y1 + 1) % board.height)),
            ("UP", dy_up, (x1, (y1 - 1) % board.height))
        ]
        
        # Filter out backwards move
        if backwards:
            direction_data = [(d, dist, pos) for d, dist, pos in direction_data if d != backwards]
        
        # Check safety for each move and filter to only safe moves
        safe_moves = []
        unsafe_moves = []
        
        for direction, distance, next_pos in direction_data:
            # Check if position is safe (not on a trail, except opponent's head)
            is_safe = (board.get_cell_state(next_pos) == EMPTY) or (next_pos == target_pos)
            
            if is_safe:
                safe_moves.append((direction, distance))
            else:
                unsafe_moves.append((direction, distance))
        
        # Sort safe moves by distance (shortest first)
        safe_moves.sort(key=lambda x: x[1])
        
        # Return the safest move with shortest distance, or any move if no safe ones
        if safe_moves:
            return safe_moves[0][0]
        elif unsafe_moves:
            # If no safe moves, sort unsafe by distance and pick shortest
            unsafe_moves.sort(key=lambda x: x[1])
            return unsafe_moves[0][0]
        else:
            return "UP"  # Fallback
    
    move_str = get_direction_to_target(my_pos, opponent_pos, board, my_agent.trail)
    
    # Convert to Direction enum
    direction_map = {
        "UP": Direction.UP,
        "DOWN": Direction.DOWN,
        "LEFT": Direction.LEFT,
        "RIGHT": Direction.RIGHT
    }
    
    # Calculate distance for boost decision
    distance_to_opponent = torus_distance(my_pos, opponent_pos, board)
    
    # Use boost aggressively when close to opponent
    use_boost = my_agent.boosts_remaining > 0 and distance_to_opponent < 5

    print(direction_map[move_str], use_boost)
    
    return direction_map[move_str], use_boost

# USAGE INSTRUCTIONS:
# ====================
# To use this in local_judge.py:
#
# 1. Import at the top of local_judge.py:
#    import local_ram
#
# 2. Replace the send_move_agent1() method with:
#    def send_move_agent1(self):
#        return local_ram.send_move_agent1(self.game)
#
# Or call it directly:
#    dir1, boost1 = local_ram.send_move_agent1(judge.game)
#
# The function expects a Game object with:
#   - game.agent1 (Agent object with .trail, .boosts_remaining)
#   - game.agent2 (Agent object with .trail)
#   - game.board (GameBoard object with .width, .height, .get_cell_state())
#
# The function returns:
#   - (Direction, bool) tuple where Direction is a Direction enum and bool indicates boost usage
