# ✅ Reward System Updated - Training Visualization Added

## Detailed Reward Penalty System

The RL agent now uses your exact reward structure:

### 🏆 When RL Agent WINS:
- **+25 points**: Opponent crosses RL's trail
- **+10 points**: Opponent crosses its own trail

### ❌ When RL Agent LOSES:
- **-10 points**: RL crosses opponent's trail  
- **-25 points**: RL crosses its own trail

### 🤝 Draw:
- **0 points**: Both agents crash or max turns reached

---

## 📊 Training Visualization

The training script now includes **real-time visualization**!

### Usage:

```bash
python train_rl_agent.py
```

When prompted, choose visualization frequency:
- **0**: No visualization (fastest training, ~30 sec for 500 episodes)
- **1**: Visualize final episode only
- **10**: Visualize every 10th episode (recommended for monitoring)
- **50**: Visualize every 50th episode (periodic check-ins)

### What You'll See:

```
==========================================
TRAINING Episode 10 | Turn: 45
Heuristic (H/Green): 67 trail, 1 boosts
RL Agent  (R/Cyan):  54 trail, 2 boosts | ε=0.850
==========================================
|...█████.........H......|
|...█...█................|
|...█...█................|
|...█████................|
|.........R█████.........|
|..........█...█.........|
==========================================

🎉 RL AGENT WINS!
Reason: Opponent crossed RL's path (+25)
Final Score: 15.3
```

### Color Coding:
- **Green (H)**: Heuristic agent
- **Cyan (R)**: RL agent (you!)
- **ε value**: Epsilon (exploration rate) - watch it decay from 1.0 → 0.1

---

## 📈 Training Output

Every 10 episodes you'll see:

```
Episode   10 | LOSS | Turns:  45 | Score: -15.3 | Epsilon: 0.850 | Win Rate: 10.00% | Avg Score: -12.5
  → RL crossed opponent's path (-10)
  
Episode   20 | WIN  | Turns:  67 | Score:  15.2 | Epsilon: 0.723 | Win Rate: 15.00% | Avg Score:  -8.2
  → Opponent crossed RL's path (+25)
```

Every 50 episodes (model save):

```
  → Model saved at episode 50 (elapsed: 45.2s)
  → Reward distribution: +25=8, +10=3, 0=2, -10=15, -25=22
```

Final summary shows complete reward breakdown:

```
Final Reward Distribution:
  +25 (Opponent hit RL's trail):   89 ( 17.8%)
  +10 (Opponent hit own trail):    12 (  2.4%)
    0 (Draw):                        5 (  1.0%)
  -10 (RL hit opponent's trail):  145 ( 29.0%)
  -25 (RL hit own trail):         249 ( 49.8%)
```

---

## 🎯 What to Watch For

### Early Training (Episodes 1-100):
- High **-25** count (RL crashes into itself a lot)
- Low win rate (~5-10%)
- Random-looking movements
- Epsilon near 1.0 (exploring)

### Mid Training (Episodes 100-300):
- **-25** decreases, **-10** increases (learning to avoid self, but still hits opponent)
- Win rate climbs to ~15-25%
- More purposeful movements
- Epsilon ~0.5-0.7

### Late Training (Episodes 300-500):
- **+25** starts increasing (forcing opponent into traps!)
- Win rate ~25-40%
- Strategic play visible
- Epsilon ~0.1-0.3 (mostly exploiting learned strategies)

---

## 💡 Tips

**Fast training (no visualization):**
```bash
# Just press Enter when prompted
python train_rl_agent.py
[Enter]
```

**Watch key episodes:**
```bash
# Visualize every 50 episodes
python train_rl_agent.py
50
```

**Debug early learning:**
```bash
# Visualize every episode (slow but informative)
python train_rl_agent.py
1
```

**Interrupt anytime:** Press `Ctrl+C` - model auto-saves!

---

## 🚀 Next Steps

1. Train for 500 episodes (~20-30 min without visualization)
2. Watch reward distribution shift from -25/-10 → +10/+25
3. Test trained model: `python local_judge.py`
4. Compare: Untrained RL vs Trained RL vs Heuristic

The visualization lets you see exactly when and how the RL agent learns to survive, then to compete, then to WIN! 🎮
