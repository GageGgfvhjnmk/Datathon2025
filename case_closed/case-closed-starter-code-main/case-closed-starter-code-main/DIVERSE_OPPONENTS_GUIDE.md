# 🎮 Diverse Training Opponents - Complete Guide

## Overview

Your RL agent can now train against **21 different opponent types** instead of just 9!

**Total Opponents:**
- 4 Basic strategies
- 5 TronBot algorithms  
- **12 NEW Diverse opponents** ✨

## 🆕 New Diverse Opponents

### 🔴 AGGRESSIVE (2 types)

**1. Berserker**
- Ultra-aggressive hunter
- **Always** chases opponent directly
- **Always** uses boost when available
- High-risk, high-reward playstyle
- Good for training against relentless pressure

**2. Hunter**
- Smart aggressive chaser
- Strategic boost usage (only when close to opponent)
- Boosts when distance < 5 cells
- More calculated than Berserker
- Good for training against tactical aggression

### 🔵 DEFENSIVE (2 types)

**3. Turtle**
- Ultra-defensive retreater
- **Always** runs away from opponent
- **Never** uses boost (saves them all)
- Very conservative playstyle
- Good for training against evasive opponents

**4. CornerCamper**
- Seeks out corners of the board
- Stays in corners for safety
- Very defensive positioning
- Avoids opponent contact
- Good for training against positional play

### 🟢 TERRITORIAL (2 types)

**5. TerritoryClaimer**
- Makes large spiral loops
- Tries to enclose big areas
- Claims territory systematically
- Occasional boost usage (every 20 turns)
- Good for training against space control

**6. EdgeRunner**
- Runs along board edges
- Claims perimeter territory
- Follows the walls
- Predictable movement pattern
- Good for training against edge-based strategies

### 🟡 UNPREDICTABLE (2 types)

**7. Chaos**
- Completely random moves
- Random boost usage (30% chance)
- No strategy whatsoever
- Highly unpredictable
- Good for training against randomness

**8. Unpredictable**
- Randomly switches strategies each turn
- Could be aggressive, defensive, territorial, or chaotic
- Different behavior every move
- Impossible to predict
- Good for training adaptability

### 🟣 SMART (2 types)

**9. SpaceMaximizer**
- Evaluates all directions
- Chooses move with most available space
- Uses BFS to count reachable cells (depth 10)
- Smart boost: only when space is tight (< 20 cells)
- Good for training against intelligent space management

**10. Cutoff**
- Predicts opponent's next move
- Tries to cut off opponent's path
- Blocks escape routes
- Boosts when close (distance < 8)
- Good for training against predictive play

### 🟠 BOOST-FOCUSED (2 types)

**11. BoostHoarder**
- **Never** uses boost
- Saves all boosts for emergency (but never uses them!)
- Conservative move selection
- Focuses on safest paths
- Good for training against boost conservation

**12. BoostSpammer**
- **Always** uses boost when available
- Extremely aggressive boost usage
- Burns through boosts quickly
- Fast and unpredictable
- Good for training against boost-heavy opponents

## 🚀 How to Use

### Default Training (All 21 Opponents)
```bash
python integrated_train_rl.py --episodes 1000
```

### Training WITHOUT Diverse Opponents (Original 9)
```bash
python integrated_train_rl.py --episodes 1000 --no-diverse
```

### Quick Test (5 Episodes)
```bash
python integrated_train_rl.py --episodes 5
```

### Extended Training (10,000 Episodes)
```bash
python integrated_train_rl.py --episodes 10000 --save-interval 500
```

## 📊 Training Benefits

**Why Train Against Diverse Opponents?**

1. **Variety of Playstyles**: Experience aggressive, defensive, territorial, and unpredictable opponents
2. **Better Generalization**: Learn to handle any type of opponent strategy
3. **Robust Strategy**: Don't overfit to just one opponent type
4. **Real-World Readiness**: Face opponents similar to what you'll see in competition
5. **Identify Weaknesses**: See which opponent types give your agent trouble

## 🎯 Progressive Difficulty

The training system still uses progressive difficulty:

- **🟢 Basic (0-20%)**: Simple opponents (LocalRAM, Random, etc.)
- **🟡 Novice (20-50%)**: Mix of basic + simple diverse opponents
- **🟠 Intermediate (50-80%)**: More diverse + TronBot algorithms
- **🔴 Expert (80-100%)**: All opponents including hardest diverse strategies

## 📈 Expected Win Rates

**Against Different Opponent Types:**

- **Easy**: Random, Chaos, BoostHoarder - Expected 80-95% win rate
- **Medium**: Turtle, EdgeRunner, TerritoryClaimer - Expected 60-80% win rate
- **Hard**: Hunter, SpaceMaximizer, Cutoff - Expected 50-70% win rate
- **Very Hard**: Berserker, Unpredictable - Expected 40-60% win rate

Your agent should achieve **60-75% overall win rate** after 5000 episodes against all 21 opponents.

## 🔍 Opponent Statistics

After training, you'll see detailed stats like:

```
👥 Opponent Performance Summary:
   Berserker       45.2% win rate ( 23- 28-  0)
   BoostHoarder    92.3% win rate ( 12-  1-  0)
   Chaos           88.9% win rate ( 16-  2-  0)
   Cutoff          58.3% win rate ( 14- 10-  0)
   Hunter          51.7% win rate ( 15- 14-  0)
   ...
```

This shows which opponents are hardest for your agent!

## 🛠️ Customization

**Want to add your own opponent?**

Edit `diverse_opponents.py` and add your strategy:

```python
@staticmethod
def my_custom_bot(game, my_agent, opponent):
    # Your strategy here
    direction = Direction.RIGHT  # Choose direction
    use_boost = False  # Choose whether to boost
    return direction, use_boost
```

Then add it to `get_diverse_opponents()`:

```python
("MyCustomBot", opponents.my_custom_bot),
```

## 📝 Summary

You now have:
- ✅ **12 new diverse opponents**
- ✅ **21 total opponent types**
- ✅ **6 categories of strategies**
- ✅ **Progressive difficulty training**
- ✅ **Detailed opponent statistics**

Train your agent against this diverse set to create a **robust, adaptable champion!** 🏆
