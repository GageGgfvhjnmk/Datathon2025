"""
Diverse Opponent Strategies for RL Training
A collection of different opponent types to train against
"""

import random
import math
from collections import deque
from case_closed_game import Direction


class DiverseOpponents:
    """Collection of diverse opponent strategies"""
    
    def __init__(self):
        pass
    
    # ==================== AGGRESSIVE OPPONENTS ====================
    
    @staticmethod
    def berserker_bot(game, my_agent, opponent):
        """
        Ultra-aggressive - always moves directly toward opponent
        Uses boost whenever possible
        """
        my_x, my_y = my_agent.trail[-1]
        opp_x, opp_y = opponent.trail[-1]
        
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Calculate direction to opponent with torus wrapping
        dx = opp_x - my_x
        dy = opp_y - my_y
        
        if abs(dx) > width // 2:
            dx = -dx / abs(dx) * (width - abs(dx))
        if abs(dy) > height // 2:
            dy = -dy / abs(dy) * (height - abs(dy))
        
        # Choose direction toward opponent
        if abs(dx) > abs(dy):
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy > 0 else Direction.UP
        
        # Check if safe
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            # Not safe, pick any safe move
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        # Always try to boost
        use_boost = my_agent.boosts_remaining > 0
        return direction, use_boost
    
    @staticmethod
    def hunter_bot(game, my_agent, opponent):
        """
        Aggressive hunter - chases opponent but preserves boosts
        Only boosts when very close to opponent
        """
        my_x, my_y = my_agent.trail[-1]
        opp_x, opp_y = opponent.trail[-1]
        
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Calculate torus distance
        dx = abs(opp_x - my_x)
        dy = abs(opp_y - my_y)
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        distance = dx + dy
        
        # Move toward opponent
        dx_raw = opp_x - my_x
        dy_raw = opp_y - my_y
        
        if abs(dx_raw) > width // 2:
            dx_raw = -dx_raw / abs(dx_raw) * (width - abs(dx_raw))
        if abs(dy_raw) > height // 2:
            dy_raw = -dy_raw / abs(dy_raw) * (height - abs(dy_raw))
        
        if abs(dx_raw) > abs(dy_raw):
            direction = Direction.RIGHT if dx_raw > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy_raw > 0 else Direction.UP
        
        # Validate safety
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        # Only boost when close (distance < 5)
        use_boost = my_agent.boosts_remaining > 0 and distance < 5
        return direction, use_boost
    
    # ==================== DEFENSIVE OPPONENTS ====================
    
    @staticmethod
    def turtle_bot(game, my_agent, opponent):
        """
        Ultra-defensive - always moves away from opponent
        Never uses boost (saves it)
        """
        my_x, my_y = my_agent.trail[-1]
        opp_x, opp_y = opponent.trail[-1]
        
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Calculate direction away from opponent
        dx = my_x - opp_x
        dy = my_y - opp_y
        
        if abs(dx) > width // 2:
            dx = -dx / abs(dx) * (width - abs(dx))
        if abs(dy) > height // 2:
            dy = -dy / abs(dy) * (height - abs(dy))
        
        # Move away
        if abs(dx) > abs(dy):
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy > 0 else Direction.UP
        
        # Validate
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        return direction, False  # Never boost
    
    @staticmethod
    def corner_camper_bot(game, my_agent, opponent):
        """
        Tries to reach and stay in corners
        Very defensive, avoids opponent
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Find nearest corner
        corners = [(0, 0), (0, height-1), (width-1, 0), (width-1, height-1)]
        nearest_corner = min(corners, key=lambda c: abs(c[0]-my_x) + abs(c[1]-my_y))
        
        # Move toward corner
        dx = nearest_corner[0] - my_x
        dy = nearest_corner[1] - my_y
        
        if abs(dx) > abs(dy) and dx != 0:
            direction = Direction.RIGHT if dx > 0 else Direction.LEFT
        elif dy != 0:
            direction = Direction.DOWN if dy > 0 else Direction.UP
        else:
            # In corner, just pick safe move
            direction = Direction.RIGHT
        
        # Validate
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        return direction, False
    
    # ==================== TERRITORIAL OPPONENTS ====================
    
    @staticmethod
    def territory_claimer_bot(game, my_agent, opponent):
        """
        Claims territory by making large loops
        Tries to enclose large areas
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Simple spiral pattern to claim territory
        # Use game turn count to determine movement pattern
        turn = len(my_agent.trail)
        pattern_length = 10
        position_in_pattern = turn % (pattern_length * 4)
        
        if position_in_pattern < pattern_length:
            direction = Direction.RIGHT
        elif position_in_pattern < pattern_length * 2:
            direction = Direction.DOWN
        elif position_in_pattern < pattern_length * 3:
            direction = Direction.LEFT
        else:
            direction = Direction.UP
        
        # Validate safety
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            # Try other directions
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        # Boost occasionally to claim more territory
        use_boost = my_agent.boosts_remaining > 0 and turn % 20 == 0
        return direction, use_boost
    
    @staticmethod
    def edge_runner_bot(game, my_agent, opponent):
        """
        Runs along the edges of the board
        Claims perimeter territory
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Determine if we're on an edge
        on_top = my_y == 0
        on_bottom = my_y == height - 1
        on_left = my_x == 0
        on_right = my_x == width - 1
        
        # Try to move along edges
        if on_top and not on_right:
            direction = Direction.RIGHT
        elif on_right and not on_bottom:
            direction = Direction.DOWN
        elif on_bottom and not on_left:
            direction = Direction.LEFT
        elif on_left and not on_top:
            direction = Direction.UP
        else:
            # Move toward nearest edge
            dist_to_edges = [my_y, height-1-my_y, my_x, width-1-my_x]
            nearest_edge = dist_to_edges.index(min(dist_to_edges))
            direction = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT][nearest_edge]
        
        # Validate
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        return direction, False
    
    # ==================== UNPREDICTABLE OPPONENTS ====================
    
    @staticmethod
    def chaos_bot(game, my_agent, opponent):
        """
        Completely unpredictable - random safe moves
        Random boost usage
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Get all safe moves
        safe_moves = []
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            tx = (my_x + d.value[0]) % width
            ty = (my_y + d.value[1]) % height
            if game.board.grid[ty][tx] == 0:
                safe_moves.append(d)
        
        if safe_moves:
            direction = random.choice(safe_moves)
        else:
            direction = Direction.RIGHT
        
        # Random boost
        use_boost = my_agent.boosts_remaining > 0 and random.random() < 0.3
        return direction, use_boost
    
    @staticmethod
    def unpredictable_bot(game, my_agent, opponent):
        """
        Mixes strategies randomly each turn
        Sometimes aggressive, sometimes defensive
        """
        strategies = [
            DiverseOpponents.berserker_bot,
            DiverseOpponents.turtle_bot,
            DiverseOpponents.territory_claimer_bot,
            DiverseOpponents.chaos_bot
        ]
        
        # Pick random strategy for this turn
        strategy = random.choice(strategies)
        return strategy(game, my_agent, opponent)
    
    # ==================== SMART OPPONENTS ====================
    
    @staticmethod
    def space_maximizer_bot(game, my_agent, opponent):
        """
        Chooses move that maximizes available space
        Smart but computationally simple
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        best_direction = Direction.RIGHT
        max_space = 0
        
        # Evaluate each direction
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            tx = (my_x + direction.value[0]) % width
            ty = (my_y + direction.value[1]) % height
            
            if game.board.grid[ty][tx] != 0:
                continue
            
            # Count reachable space from this position
            space = DiverseOpponents._count_reachable_space(game, (tx, ty), max_depth=10)
            
            if space > max_space:
                max_space = space
                best_direction = direction
        
        # Smart boost: only when space is tight
        use_boost = my_agent.boosts_remaining > 0 and max_space < 20
        return best_direction, use_boost
    
    @staticmethod
    def _count_reachable_space(game, start_pos, max_depth=10):
        """Helper: Count reachable empty cells using BFS"""
        visited = set()
        queue = deque([(start_pos, 0)])
        count = 0
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        while queue:
            (x, y), depth = queue.popleft()
            
            if (x, y) in visited or depth > max_depth:
                continue
            
            if game.board.grid[y][x] != 0:
                continue
            
            visited.add((x, y))
            count += 1
            
            # Add neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = (x + dx) % width
                ny = (y + dy) % height
                if (nx, ny) not in visited:
                    queue.append(((nx, ny), depth + 1))
        
        return count
    
    @staticmethod
    def cutoff_bot(game, my_agent, opponent):
        """
        Tries to cut off opponent's path
        Predicts opponent movement and blocks
        """
        my_x, my_y = my_agent.trail[-1]
        opp_x, opp_y = opponent.trail[-1]
        
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Predict opponent's next move (they're likely moving away from us)
        dx = opp_x - my_x
        dy = opp_y - my_y
        
        if abs(dx) > width // 2:
            dx = -dx / abs(dx) * (width - abs(dx))
        if abs(dy) > height // 2:
            dy = -dy / abs(dy) * (height - abs(dy))
        
        # Predict opponent will continue in same direction
        predicted_opp_dir = opponent.direction
        pred_x = (opp_x + predicted_opp_dir.value[0]) % width
        pred_y = (opp_y + predicted_opp_dir.value[1]) % height
        
        # Try to move toward predicted position
        dx_to_pred = pred_x - my_x
        dy_to_pred = pred_y - my_y
        
        if abs(dx_to_pred) > abs(dy_to_pred):
            direction = Direction.RIGHT if dx_to_pred > 0 else Direction.LEFT
        else:
            direction = Direction.DOWN if dy_to_pred > 0 else Direction.UP
        
        # Validate
        test_x = (my_x + direction.value[0]) % width
        test_y = (my_y + direction.value[1]) % height
        
        if game.board.grid[test_y][test_x] != 0:
            for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                tx = (my_x + d.value[0]) % width
                ty = (my_y + d.value[1]) % height
                if game.board.grid[ty][tx] == 0:
                    direction = d
                    break
        
        # Boost when close to opponent
        distance = abs(dx) + abs(dy)
        use_boost = my_agent.boosts_remaining > 0 and distance < 8
        return direction, use_boost
    
    # ==================== BOOST-FOCUSED OPPONENTS ====================
    
    @staticmethod
    def boost_hoarder_bot(game, my_agent, opponent):
        """
        Never uses boost - saves all boosts
        Conservative playstyle
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Just pick the safest-looking direction
        best_direction = Direction.RIGHT
        max_exits = 0
        
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            tx = (my_x + direction.value[0]) % width
            ty = (my_y + direction.value[1]) % height
            
            if game.board.grid[ty][tx] != 0:
                continue
            
            # Count exits from this position
            exits = 0
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx = (tx + dx) % width
                ny = (ty + dy) % height
                if game.board.grid[ny][nx] == 0:
                    exits += 1
            
            if exits > max_exits:
                max_exits = exits
                best_direction = direction
        
        return best_direction, False  # NEVER boost
    
    @staticmethod
    def boost_spammer_bot(game, my_agent, opponent):
        """
        Uses boost at every opportunity
        Aggressive and fast
        """
        my_x, my_y = my_agent.trail[-1]
        width = len(game.board.grid[0])
        height = len(game.board.grid)
        
        # Pick random safe direction
        safe_moves = []
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            tx = (my_x + d.value[0]) % width
            ty = (my_y + d.value[1]) % height
            if game.board.grid[ty][tx] == 0:
                safe_moves.append(d)
        
        direction = random.choice(safe_moves) if safe_moves else Direction.RIGHT
        
        # ALWAYS boost if available
        use_boost = my_agent.boosts_remaining > 0
        return direction, use_boost


def get_diverse_opponents():
    """
    Get list of all diverse opponent strategies
    Returns: List of (name, function) tuples
    """
    opponents = DiverseOpponents()
    
    return [
        # Aggressive
        ("Berserker", opponents.berserker_bot),
        ("Hunter", opponents.hunter_bot),
        
        # Defensive
        ("Turtle", opponents.turtle_bot),
        ("CornerCamper", opponents.corner_camper_bot),
        
        # Territorial
        ("TerritoryClaimer", opponents.territory_claimer_bot),
        ("EdgeRunner", opponents.edge_runner_bot),
        
        # Unpredictable
        ("Chaos", opponents.chaos_bot),
        ("Unpredictable", opponents.unpredictable_bot),
        
        # Smart
        ("SpaceMaximizer", opponents.space_maximizer_bot),
        ("Cutoff", opponents.cutoff_bot),
        
        # Boost-focused
        ("BoostHoarder", opponents.boost_hoarder_bot),
        ("BoostSpammer", opponents.boost_spammer_bot),
    ]


if __name__ == "__main__":
    # Test: print all available opponents
    opponents = get_diverse_opponents()
    print("=" * 60)
    print("DIVERSE OPPONENT STRATEGIES")
    print("=" * 60)
    print(f"\nTotal opponents: {len(opponents)}\n")
    
    print("🔴 AGGRESSIVE:")
    print("  - Berserker: Ultra-aggressive, always chases, always boosts")
    print("  - Hunter: Aggressive chaser, strategic boost usage")
    
    print("\n🔵 DEFENSIVE:")
    print("  - Turtle: Runs away, never boosts")
    print("  - CornerCamper: Seeks corners, very defensive")
    
    print("\n🟢 TERRITORIAL:")
    print("  - TerritoryClaimer: Makes loops to claim large areas")
    print("  - EdgeRunner: Runs along board edges")
    
    print("\n🟡 UNPREDICTABLE:")
    print("  - Chaos: Completely random moves and boosts")
    print("  - Unpredictable: Randomly switches strategies each turn")
    
    print("\n🟣 SMART:")
    print("  - SpaceMaximizer: Chooses move with most available space")
    print("  - Cutoff: Predicts opponent movement and cuts them off")
    
    print("\n🟠 BOOST-FOCUSED:")
    print("  - BoostHoarder: Never uses boost (saves all)")
    print("  - BoostSpammer: Uses boost every opportunity")
    
    print("\n" + "=" * 60)
