"""Detailed test to verify RL agent's hard-coded self-collision prevention"""
from local_judge import LocalJudge
import sys
from io import StringIO

print("Testing RL Agent Self-Collision Prevention (Detailed)...")
print("Running 20 games to check safety implementation\n")

self_collision_trapped = 0  # Trapped scenarios (acceptable - will improve with training)
self_collision_avoidable = 0  # Should NEVER happen - bug in safety code
opponent_collision_count = 0
rl_wins = 0
total_games = 20

for i in range(total_games):
    # Capture output to detect "Truly trapped" message
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    judge = LocalJudge()
    result = judge.run_game(delay=0.0, visualize=False)
    
    output = captured_output.getvalue()
    sys.stdout = old_stdout
    
    # Print the captured output
    print(output, end='')
    
    # Check if it was a "truly trapped" scenario
    was_trapped = "Truly trapped!" in output
    had_emergency = "WARNING: No safe candidates found!" in output
    
    # Check result
    # POSITIVE score = Agent 1 wins (RL loses)
    # NEGATIVE score = Agent 2 (RL) wins
    
    if result == 10:  # Agent 1 won because Agent 2 crossed itself
        if was_trapped:
            self_collision_trapped += 1
            print(f"Game {i+1}: ⚠️ RL crossed itself (TRAPPED - acceptable)")
        else:
            self_collision_avoidable += 1
            print(f"Game {i+1}: ❌ RL CROSSED ITSELF (AVOIDABLE - BUG!)")
    elif result == 25:  # Agent 1 won because Agent 2 hit Agent 1's trail (backwards or opponent trail)
        rl_wins += 1
        print(f"Game {i+1}: ✅ RL won (opponent hit RL)")
    elif result == -10:  # Agent 2 (RL) won because Agent 1 crossed RL's trail
        opponent_collision_count += 1
        rl_wins += 1
        print(f"Game {i+1}: ✅ RL won (hit opponent)")
    elif result == -25:  # Agent 2 (RL) won because Agent 1 crossed its own trail
        rl_wins += 1
        print(f"Game {i+1}: ✅ RL won (opponent crossed itself)")
    else:
        print(f"Game {i+1}: Draw")

print("\n" + "="*60)
print(f"RESULTS:")
print(f"  RL Wins: {rl_wins}/{total_games}")
print(f"  Self-collisions (TRAPPED): {self_collision_trapped}/{total_games}")
print(f"  Self-collisions (AVOIDABLE): {self_collision_avoidable}/{total_games}")
print(f"  Opponent-collisions: {opponent_collision_count}/{total_games}")
print()

if self_collision_avoidable == 0:
    print("  ✅ SUCCESS! Hard-coded safety is working correctly!")
    print("  The RL agent never made avoidable self-collisions.")
    if self_collision_trapped > 0:
        print(f"  ⚠️ {self_collision_trapped} trapped scenarios - RL will learn to avoid these through training")
else:
    print(f"  ❌ FAILURE! RL had {self_collision_avoidable} AVOIDABLE self-collisions")
    print("  This indicates a bug in the hard-coded safety logic.")
print("="*60)
