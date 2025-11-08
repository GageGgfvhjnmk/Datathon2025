from agent import *

def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining

    #Q-table
    #Each state will be represented as a string of the board + agent positions
    #Actions will be UP, DOWN, LEFT, RIGHT, and optionally BOOST
    #Action a in state s
    #Table is agent's memory and strategy

    #look up q(s,a) values
    #Choose action with highest value
    #Occasionally explore random action
    #Over time, exploit more, explore less

    #Training loop - observe current state, choose a, receive reward, observe next state, update q(s,a)
    #Repeat for many episodes to learn optimal strategy

    #Bellman Equation - update q(s,a) based on reward and max q(s',a') for next state s'
    #a is learning rate
    #Q(s,a) = Q(s,a) + a * (reward + y * max_a' Q(s',a') - Q(s,a))
    #y is discount factor for future rewards
    #Helps agent learn long-term strategies
    #How much should the next be considered

    #Survive as long as possible
    #Rewards - Death Penalty (-100), Survival Reward (+1 per move), Opponent Death Reward (+50)
    #States: (danger_left, danger_right, danger_up, danger_down)
    #Agent is short-sighted - only immediate dangers

    #More advanced state - open space can be considered
    #DQN - for better satte
    #Can use BFS/DFS to find out how much space

    def generate_report():
        os.system('cls')
        board = state['board'][:]
        report = {}
        report['my_position'] = my_agent.trail[-1]
        report['my_trail'] = list(my_agent.trail)
        report['my_length'] = my_agent.length
        report['my_boosts'] = my_agent.boosts_remaining
        #Find my position
        if list(report['my_position']) == state['agent1_trail'][-1]:
            my_x, my_y = state['agent1_trail'][-1]
            opp_x, opp_y = state['agent2_trail'][-1]
            for y in range(len(board)):
                for x in range(len(board[y])):
                    if (x, y) == (my_x, my_y):
                        board[y][x] = '\033[92mA\033[0m'  # Mark my trail on the board for visualization
                    elif (x, y) == (opp_x, opp_y):
                        board[y][x] = '\033[91mB\033[0m'  # Mark opponent's trail on the board for visualization
                    elif [x,y] in state['agent1_trail'][:-1]:
                        board[y][x] = '\033[92m█\033[0m'  # Mark my trail on the board for visualization
                    elif [x,y] in state['agent2_trail'][:-1]:
                        board[y][x] = '\033[91m█\033[0m'  # Mark opponent's trail on the board for visualization
                    else:
                        board[y][x] = '\033[90m█\033[0m'
            
            # in state['agent1_trail'][:-1]:
            #     x, y = cell
            #     board[y][x] = '\033[92m█\033[0m'  # Mark my trail on the board for visualization
            # for cell in state['agent2_trail'][:-1]:
            #     x, y = cell
            #     board[y][x] = '\033[91m█\033[0m'  # Mark opponent's trail on the board for visualization
            # for cell in 
        else:
            my_x, my_y = state['agent2_trail'][-1]
            opp_x, opp_y = state['agent1_trail'][-1]
            for cell in state['agent2_trail'][:-1]:
                x, y = cell
                board[y][x] = '\033[92m█\033[0m'  # Mark my trail on the board for visualization
            for cell in state['agent1_trail'][:-1]:
                x, y = cell
                board[y][x] = '\033[91m█\033[0m'  # Mark opponent's trail on the board for visualization
        print(f"My Position: {report['my_position']}, Opponent Position: {(opp_x, opp_y)}")
        board[my_y][my_x] = f'\033[92mA\033[0m'  # Mark my position on the board for visualization
        board[opp_y][opp_x] = f'\033[91mB\033[0m'  # Mark my position on the board for visualization
        print("Board State:")
        for row in board:
            string = ''.join(str(cell) for cell in row)
            print(string)
        

    generate_report()

    # -----------------your code here-------------------
    # Simple example: always go RIGHT (replace this with your logic)
    # To use a boost: move = "RIGHT:BOOST"
    # Level 1: The rammer
    # Detect opponent position
    # Run into the opponent with the shortest path possible (preventing the wall built by the opponent)

    # Get agent positions
    opponent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
    my_pos = my_agent.trail[-1]  # My current head position
    opponent_pos = opponent.trail[-1]  # Opponent's head position
    
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
        
        print(f"Opponent Position: {opponent_pos}, My Position: {my_pos}")
        print(f"  dx_left={dx_left}, dx_right={dx_right}, dy_up={dy_up}, dy_down={dy_down}")
        print(f"  Min distances: dx={dx}, dy={dy}")
        
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
        
        print(f"  Current direction: {current_direction}, Backwards (filtered): {backwards}")
        print(f"  Direction distances: RIGHT={dx_right}, LEFT={dx_left}, DOWN={dy_down}, UP={dy_up}")
        print(f"  Safe moves (sorted): {[(d, dist) for d, dist in safe_moves]}")
        print(f"  Unsafe moves: {[(d, dist) for d, dist in unsafe_moves]}")
        
        # Return the safest move with shortest distance, or any move if no safe ones
        if safe_moves:
            return safe_moves[0][0]
        elif unsafe_moves:
            # If no safe moves, sort unsafe by distance and pick shortest
            unsafe_moves.sort(key=lambda x: x[1])
            print(f"  WARNING: No safe moves! Choosing least bad: {unsafe_moves[0][0]}")
            return unsafe_moves[0][0]
        else:
            return "UP"  # Fallback
    
    move = get_direction_to_target(my_pos, opponent_pos, GLOBAL_GAME.board, my_agent.trail)
    
    # Use boost aggressively when close to opponent
    distance_to_opponent = torus_distance(my_pos, opponent_pos, GLOBAL_GAME.board)
    if boosts_remaining > 0 and distance_to_opponent < 5:
        move += ":BOOST"
    
    print(f'Player {player_number}: pos={my_pos}, opponent={opponent_pos}, distance={distance_to_opponent}, move={move}, boosts={boosts_remaining}')
    print(move)
    # Example: Use boost if available and it's late in the game
    # turn_count = state.get("turn_count", 0)
    # if boosts_remaining > 0 and turn_count > 50:
    #     move = "RIGHT:BOOST"
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200
