# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Pre-commit hooks
pre-commit install
```

### Docker Development
```bash
# Build Docker image
./build.sh

# Run training in Docker (replace 0 with GPU ID, or "all" for all GPUs)
./run_docker.sh 0 "python ppo_skills_chatgpt.py"
```

### Training Commands
```bash
# Basic PPO training
python ppo.py

# Skill-based training variants
python ppo_skills.py                    # Basic skill switching
python ppo_skills_chatgpt.py           # LLM-generated rewards + meta-policy
python ppo_skills_chatgpt_diayn.py     # DIAYN + skill-based training
python ppo_diayn.py                     # Pure DIAYN diversity training

# Other RL variants
python ppo_rnn.py                       # RNN-based PPO
python ppo_rnd.py                       # Random Network Distillation
python ppo.py --train_icm               # Intrinsic Curiosity Module
python ppo.py --train_icm --use_e3b --icm_reward_coeff 0  # E3B
```

### Visualization
```bash
# View trained policies (pass path to wandb run's files directory)
python analysis/view_ppo_skills_agent.py --path wandb/run-xxx/files
```

### Wandb Sweeps
```bash
# Create and run a learning rate sweep
wandb sweep sweep_config.yaml  # Returns sweep ID
wandb agent <sweep_id>         # Run sweep agent (can run multiple agents in parallel)

# Example sweep for learning rates
wandb sweep sweep_config.yaml
# Copy the returned sweep ID (e.g., username/project/sweep_id)
wandb agent username/project/sweep_id
```

## Code Architecture

### Core Training Pipeline
The codebase implements hierarchical reinforcement learning with PPO as the base algorithm. All training scripts follow a similar pattern:
1. Environment setup with Craftax
2. Actor-critic network initialization
3. Skill system setup (if applicable)
4. Training loop with PPO updates
5. Logging and checkpointing

### Skill System (3 Skills)
- **HARVEST (ID: 0)**: Resource gathering (wood, stone, coal, iron, diamond)
- **CRAFT (ID: 1)**: Tool/weapon creation (pickaxes, swords, tables)
- **SUSTAIN (ID: 2)**: Survival maintenance (health, food, drink, energy)

Skills are selected by meta-policy in `meta_policy/skill_training.py` with priority: Sustain > Craft > Harvest.

### Neural Network Architecture
- **Actor-Critic**: Base networks in `models/actor_critic.py`
  - `ActorCritic`: Fully-connected for symbolic obs
  - `ActorCriticConv`: Convolutional for pixel obs
- **DIAYN Integration**: Combined actor-critic-discriminator in `models/diayn_ac.py`
- **Discriminator**: Standalone skill prediction network in `models/discrim.py`

### Reward Functions
Reward functions are modular and located in `reward_fns/`:
- `my_skill_rewards.py`: Hand-tuned skill rewards
- `gemini_skill_rewards.py`: LLM-generated rewards
- Each skill has dedicated reward calculation functions

### DIAYN Implementation
DIAYN (Diversity is All You Need) adds skill diversity through:
- Discriminator network predicting skill from state
- Mutual information reward: `reward = (log q(z|s) - log p(z)) * coeff`
- Typical coefficients: 0.05-0.1
- Skill vectors concatenated to observations

### Key Files
- `ppo_skills_chatgpt.py`: Main skill training with LLM rewards
- `meta_policy/skill_training.py`: Skill selection and termination logic
- `models/diayn_ac.py`: DIAYN actor-critic-discriminator
- `reward_fns/my_skill_rewards.py`: Primary reward functions
- `wrappers.py`: Environment wrappers for logging and vectorization

### Training Flow
1. Meta-policy selects skill based on current state
2. Skill ID concatenated to observation
3. Skill-conditioned policy selects action
4. Environment step with reward calculation
5. Check skill termination conditions
6. PPO update with skill-specific advantages
7. Discriminator update (DIAYN only)

### Observation Space
Craftax provides symbolic observations including:
- Inventory counts (wood, stone, coal, iron, diamond, etc.)
- Player intrinsics (health, food, drink, energy, mana)
- Spatial map information
- Crafting table proximity and availability

### Experiment Tracking
All experiments use Weights & Biases (wandb) for logging:
- Episode rewards and metrics
- Skill transition tracking
- Discriminator accuracy (DIAYN)
- Policy checkpoints saved to `wandb/run-*/files/policies/`

### Environment Wrappers
- `LogWrapper`: Episode statistics and reward tracking
- `OptimisticResetVecEnvWrapper`: Efficient vectorized resets
- `AutoResetEnvWrapper`: Automatic episode termination handling
- `BatchEnvWrapper`: Batch environment execution

## Important Notes

### Skill Switching Logic
The meta-policy uses hierarchical decision-making:
1. Check if SUSTAIN skill should activate (low health/food/drink/energy)
2. Check if CRAFT skill should activate (can craft useful items)
3. Default to HARVEST skill (resource gathering)

### DIAYN Configuration
- Discriminator operates on base environment observations only
- Skill vectors are one-hot encoded with dimension matching number of skills
- DIAYN coefficient balances diversity vs. task performance
- Discriminator loss uses cross-entropy with skill labels

### Checkpoint Management
- Policies saved every 100M steps by default
- Checkpoints stored in Orbax format
- Use `--save_policy` flag to enable checkpoint saving
- View policies with `analysis/view_ppo_skills_agent.py`