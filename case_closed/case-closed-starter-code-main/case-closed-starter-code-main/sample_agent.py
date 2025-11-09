import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import random
from generate_report import generate_report

from case_closed_game import Game, Direction, GameResult, EMPTY
import RL_agent_strategy

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "DA_AGENT"
AGENT_NAME = "RL_AGENT"

# Initialize the RL Agent once at startup
RL_AGENT = RL_agent_strategy.get_rl_agent()
RL_AGENT.epsilon = 0.0  # No exploration during competition
RL_AGENT.training_mode = False


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
                # DEBUG: Check what the judge is sending
                print(f"DEBUG [agent.py]: Received board from judge, type={type(data['board'])}")
                if hasattr(data['board'], '__len__') and len(data['board']) > 0:
                    print(f"DEBUG [agent.py]: type(data['board'][0])={type(data['board'][0])}")
                    if hasattr(data['board'][0], '__len__') and len(data['board'][0]) > 0:
                        print(f"DEBUG [agent.py]: type(data['board'][0][0])={type(data['board'][0][0])}, value={data['board'][0][0]}")
                
                GLOBAL_GAME.board.grid = data["board"]
            except Exception as e:
                print(f"DEBUG [agent.py]: Error updating board: {e}")
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
    
    Uses trained RL agent (Heuristic-Guided DQN) for decision making.
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        boosts_remaining = my_agent.boosts_remaining
        report = {
            "my_position": my_agent.trail[-1] if my_agent.trail else (0, 0)
        }
        generate_report(state, report)
    
    # Use the trained RL agent to pick the best move
    try:
        # Get the player number from state
        player_num = state.get("player_number", player_number)
        
        # Use the RL agent's select_action method
        # It expects: select_action(game, player_number, training=False)
        direction, use_boost = RL_AGENT.select_action(GLOBAL_GAME, player_num, False)
        
        # Convert Direction enum to string
        direction_str = direction.name  # This gives "UP", "DOWN", "LEFT", "RIGHT"
        
        # Format the move
        if use_boost:
            move = f"{direction_str}:BOOST"
        else:
            move = direction_str
        
        return jsonify({"move": move}), 200
        
    except Exception as e:
        # Fallback to a safe random move if RL agent fails
        print(f"RL Agent error: {e}, falling back to random safe move")
        DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # Get current position
        player_num = state.get("player_number", player_number)
        my_trail = state.get(f"agent{player_num}_trail", [])
        if not my_trail:
            move = random.choice(DIRECTIONS)
            return jsonify({"move": move}), 200
        
        my_pos = tuple(my_trail[-1])
        board = state.get("board", [])
        width = len(board[0]) if board else 25
        height = len(board) if board else 25
        
        # Find a safe move (avoid collisions)
        safe_moves = []
        for direction in DIRECTIONS:
            dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[direction]
            nx, ny = my_pos[0] + dx, my_pos[1] + dy
            
            if 0 <= nx < width and 0 <= ny < height:
                if board[ny][nx] == 0:  # EMPTY cell
                    safe_moves.append(direction)
        
        if safe_moves:
            move = random.choice(safe_moves)
        else:
            move = random.choice(DIRECTIONS)
        
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
    port = int(os.environ.get("PORT", "5009"))
    app.run(host="0.0.0.0", port=port, debug=True)
