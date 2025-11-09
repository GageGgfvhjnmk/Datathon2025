# RL Agent Implementation - Heuristic-Guided DQN

## Solution: Hybrid Heuristic + Reinforcement Learning

**Approach:** Solution 2 - Heuristic-Guided DQN (Fastest Learning)

### Architecture

1. **Heuristic evaluates all moves** → ranks top-K candidates (K=5)
2. **DQN network** learns Q-values for each candidate
3. **Epsilon-greedy** exploration within top-K
4. **Self-play training** against heuristic agent

### Why This Is Fastest

- ✅ **Tiny action space** - Only 5 candidates instead of all moves
- ✅ **Guided exploration** - Heuristic filters bad moves
- ✅ **Bootstrap performance** - Starts at heuristic level
- ✅ **Fast convergence** - Learning "pick best from good" is easier

## Files Created

### 1. `RL_agent_strategy.py`
Main RL agent implementation:
- `DQNNetwork` - Neural network for Q-value estimation
- `HeuristicGuidedDQNAgent` - Main agent class
- `send_move_rl_agent()` - Interface for game

**Features:**
- Convolutional layers for board state
- Feature extraction from heuristic evaluations
- Experience replay memory
- Target network for stable training
- Model save/load functionality

### 2. `train_rl_agent.py`
Training script for self-play:
- Runs episodes of RL agent vs heuristic
- Tracks win rate and scores
- Saves models periodically
- Handles interruption gracefully

### 3. `local_judge.py` (Modified)
- Agent 1: Hyper-optimized heuristic
- Agent 2: RL agent (heuristic-guided DQN)

## Usage

### Play Game (RL vs Heuristic)
```bash
python local_judge.py
```

### Train RL Agent
```bash
python train_rl_agent.py
```

Training will:
- Run 500 episodes by default
- Save model every 50 episodes
- Display win rate and epsilon decay
- Save final model as `rl_agent_final.pth`

### Monitor Training
Watch for:
- **Win Rate** - Should increase over time as RL learns
- **Epsilon** - Decays from 1.0 to 0.1 (exploration → exploitation)
- **Turns** - Longer games indicate better play
- **Score** - Higher scores = better performance

## Network Architecture

```
Input: Board State (3×20×18) + Candidate Features (32-dim)
    ↓
Conv Layers (3×32→32×64→64×64)
    ↓
Fully Connected (256 units) → Board representation
    ↓
Feature Processing (128 units)
    ↓
Combined (128 units)
    ↓
Output: Q-value (1 unit per candidate)
```

## Training Process

1. **Episode Loop:**
   - RL agent plays vs heuristic
   - Store (state, action, reward, next_state)
   
2. **Experience Replay:**
   - Sample random batch from memory
   - Compute Q-targets using target network
   - Update policy network

3. **Epsilon Decay:**
   - Start: ε=1.0 (100% exploration)
   - End: ε=0.1 (10% exploration)
   - Decay: 0.995 per episode

4. **Target Network Update:**
   - Every 10 training steps
   - Stabilizes learning

## Current Status

- ✅ RL agent plays valid moves (no backwards crashes)
- ✅ Integration with local_judge complete
- ✅ Training infrastructure ready
- ⏳ Model not yet trained (starts random)

## Next Steps

1. **Train the agent:**
   ```bash
   python train_rl_agent.py
   ```

2. **Monitor performance:**
   - Initial: ~0% win rate (random exploration)
   - After 100 episodes: ~10-20% (learning patterns)
   - After 500 episodes: ~30-50% (competitive)

3. **Fine-tune hyperparameters:**
   - Learning rate: 0.0005
   - Gamma (discount): 0.95
   - Batch size: 64
   - Top-K candidates: 5

4. **Advanced improvements:**
   - Prioritized experience replay
   - Double DQN
   - Dueling network architecture
   - Curriculum learning

## Expected Training Time

- **Per episode:** ~1-2 seconds
- **500 episodes:** ~15-20 minutes
- **1000 episodes:** ~30-40 minutes

## Model Performance Expectations

| Episodes | Expected Win Rate vs Heuristic | Skill Level |
|----------|-------------------------------|-------------|
| 0-50     | 0-5%                          | Random      |
| 50-200   | 5-20%                         | Beginner    |
| 200-500  | 20-40%                        | Intermediate|
| 500-1000 | 40-60%                        | Advanced    |
| 1000+    | 60-80%                        | Expert      |

Remember: The heuristic is already hyper-optimized, so even 50% win rate means the RL agent has learned significantly!
