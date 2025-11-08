import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult, EMPTY

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "YiuRammer"
AGENT_NAME = "RammerV1"


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


@app.route("/send-move", methods=["GET"])
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
    # -----------------end code here--------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5010"))
    app.run(host="0.0.0.0", port=port, debug=True)
