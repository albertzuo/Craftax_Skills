import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.envs.craftax_state import EnvState

# Reward weights for harvesting different materials

HARVESTING_REWARD_WEIGHTS = jnp.array([
    1.0,  # Wood
    1.0,  # Stone
    1.0,  # Coal
    1.0,  # Iron
    1.0, # Diamond
    0.0,  # Sapling
], dtype=jnp.float32)
HARVESTING_MULTIPLIER = 1.0

CRAFTING_MULTIPLIER = 2.0

SURVIVAL_MULTIPLIER = 0.1

# Weights applied to the reward for crafting each corresponding item.
# Higher weights encourage crafting more advanced items.
# Order: [wood_pickaxe, stone_pickaxe, iron_pickaxe, wood_sword, stone_sword, iron_sword]
CRAFTING_REWARD_WEIGHTS = jnp.array([
    1.0,  # Reward for wood_pickaxe
    1.0,  # Reward for stone_pickaxe
    1.0,  # Reward for iron_pickaxe
    1.0,  # Reward for wood_sword
    1.0,  # Reward for stone_sword
    1.0,  # Reward for iron_sword
])

@jax.jit
def my_harvesting_reward_fn_state(prev_state: EnvState, current_state: EnvState, done: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates a reward signal focused on harvesting raw materials in Craftax.

    The reward is based on the positive change in the quantity of specific raw
    materials (wood, stone, coal, iron, diamond, sapling) in the agent's
    inventory between the previous and current state.

    Args:
        prev_state: The EnvState from the previous time step.
        current_state: The EnvState from the current time step.
        done: Boolean indicating if the episode is done.

    Returns:
        A scalar JAX array representing the reward for the current step.
    """
    # Extract raw materials directly from inventory
    prev_raw_materials = jnp.array([
        prev_state.inventory.wood,
        prev_state.inventory.stone,
        prev_state.inventory.coal,
        prev_state.inventory.iron,
        prev_state.inventory.diamond,
        prev_state.inventory.sapling
    ], dtype=jnp.float32)
    
    current_raw_materials = jnp.array([
        current_state.inventory.wood,
        current_state.inventory.stone,
        current_state.inventory.coal,
        current_state.inventory.iron,
        current_state.inventory.diamond,
        current_state.inventory.sapling
    ], dtype=jnp.float32)

    # Calculate the change in counts for each raw material
    delta_materials = (current_raw_materials - prev_raw_materials) / (prev_raw_materials + 1.0)

    # Only reward positive changes (i.e., increases in resources)
    # This prevents rewarding the agent for dropping items or using them in crafting (if that were possible here).
    positive_delta_materials = jnp.maximum(0.0, delta_materials)

    # Calculate the weighted reward
    # Multiply the positive change of each resource by its corresponding weight
    weighted_reward = positive_delta_materials * HARVESTING_REWARD_WEIGHTS #* HARVESTING_MULTIPLIER

    # Sum the weighted rewards for all resources to get the final scalar reward
    total_reward = jnp.sum(weighted_reward, axis=-1)

    # Return 0 if done, otherwise return the calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    
    # Ensure the output is a scalar float32 array
    return jnp.array(reward, dtype=jnp.float32)

@jax.jit
def my_crafting_reward_fn_state(prev_state: EnvState, current_state: EnvState, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates a reward signal focused on the crafting skill in Craftax.

    This function rewards the agent based on the *increase* in the quantity
    of specific crafted items (tools and swords) in its inventory between
    two consecutive timesteps.

    Args:
        prev_state: The EnvState from the previous timestep.
        current_state: The EnvState from the current timestep.
        done: Boolean indicating if the episode is done.

    Returns:
        A scalar float32 reward value. Positive reward is given when a target
        item count increases, weighted by CRAFTING_REWARD_WEIGHTS.
    """
    # --- 1. Extract Inventory Data ---
    # Access inventory directly from state
    prev_crafted_items = jnp.array([
        prev_state.inventory.wood_pickaxe,
        prev_state.inventory.stone_pickaxe,
        prev_state.inventory.iron_pickaxe,
        prev_state.inventory.wood_sword,
        prev_state.inventory.stone_sword,
        prev_state.inventory.iron_sword
    ], dtype=jnp.int32)
    
    current_crafted_items = jnp.array([
        current_state.inventory.wood_pickaxe,
        current_state.inventory.stone_pickaxe,
        current_state.inventory.iron_pickaxe,
        current_state.inventory.wood_sword,
        current_state.inventory.stone_sword,
        current_state.inventory.iron_sword
    ], dtype=jnp.int32)

    # --- 3. Isolate Target Crafted Item Counts ---
    # Cap counts at 1 for reward calculation
    prev_crafted_item_counts = jnp.minimum(prev_crafted_items, 1)
    current_crafted_item_counts = jnp.minimum(current_crafted_items, 1)

    # --- 4. Calculate Increase in Counts ---
    # Compute the difference between current and previous counts for target items.
    delta_counts = current_crafted_item_counts - prev_crafted_item_counts

    # --- 5. Apply Weights and Sum ---
    # Multiply the increase in count for each item by its corresponding weight.
    weighted_increase = delta_counts * CRAFTING_REWARD_WEIGHTS * CRAFTING_MULTIPLIER

    # Sum the weighted increases across all target items to get the final reward.
    total_reward = jnp.sum(weighted_increase)

    # Return 0 if done, otherwise return the calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    
    # --- 6. Return Reward ---
    # Ensure the reward is a float32, a common type for RL rewards.
    return reward.astype(jnp.float32)

@jax.jit
def my_survival_reward_fn_state(prev_state: EnvState, current_state: EnvState, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates a reward signal focused on survival in the Craftax environment.

    Args:
        prev_state: The EnvState from the previous timestep.
        current_state: The EnvState from the current timestep.
        done: Boolean indicating if the episode is done.

    Returns:
        A scalar reward value (float).
    """
    reward = 0.0

    # --- Extract Player Intrinsics (Current State) ---
    # Access intrinsics directly from state
    health = current_state.player_health
    food = current_state.player_food
    drink = current_state.player_drink
    energy = current_state.player_energy
    is_sleeping = current_state.is_sleeping

    # --- Extract Previous Intrinsics ---
    prev_health = prev_state.player_health
    prev_food = prev_state.player_food
    prev_drink = prev_state.player_drink
    prev_energy = prev_state.player_energy
    intrinsic_stat_multiplier = 0.2

    # --- Mob Proximity Penalty ---
    # Get player position from state
    player_x, player_y = current_state.player_position[0], current_state.player_position[1]
    
    # Define hostile mob indices (zombie=0, skeleton=2 based on common Craftax setup)
    zombie_idx = 0
    skeleton_idx = 2
    
    # Check 3x3 area around player for hostile mobs
    # Use dynamic_slice for JAX compatibility with dynamic indices
    local_area = jax.lax.dynamic_slice(
        current_state.mob_map,
        (player_x-1, player_y-1),
        (3, 3)
    )
    
    # Penalty for being near hostile mobs
    nearby_zombies = jnp.sum(local_area == zombie_idx)
    nearby_skeletons = jnp.sum(local_area == skeleton_idx)
    
    mob_proximity_penalty = -1 * intrinsic_stat_multiplier * (nearby_zombies + nearby_skeletons)

    # === Reward Components ===

    # 1. Health & Damage Penalty
    # Penalize taking damage (decrease in health)
    reward = health - prev_health #- 0.1
    reward += intrinsic_stat_multiplier * jnp.maximum(food - prev_food, 0.0)
    reward += intrinsic_stat_multiplier * jnp.maximum(drink - prev_drink, 0.0)
    reward += intrinsic_stat_multiplier * jnp.maximum(energy - prev_energy, 0.0)
    
    #reward += mob_proximity_penalty

    # is_reset = jnp.logical_and(prev_health < 8.0, health == 9.0)
    reward = jnp.where(done, 0.0, reward * SURVIVAL_MULTIPLIER)

    return reward.astype(jnp.float32)

HARVEST_ACHIEVEMENT_REWARDS = {
    # Basic progression - essential for iron crafting
    Achievement.COLLECT_WOOD.value: {"reward": 1.0, "enabled": True},
    
    # Stone progression - needed for better mining
    Achievement.COLLECT_STONE.value: {"reward": 1.0, "enabled": True},
    
    # Iron prerequisites - furnace needed for smelting
    Achievement.COLLECT_COAL.value: {"reward": 1.0, "enabled": True},
    Achievement.COLLECT_IRON.value: {"reward": 1.0, "enabled": True},
    
    # Optional achievements (can be toggled off to find minimal set)
    Achievement.COLLECT_SAPLING.value: {"reward": 1.0, "enabled": False},
    Achievement.COLLECT_DIAMOND.value: {"reward": 1.0, "enabled": True},
}

CRAFTING_ACHIEVEMENT_REWARDS = {
    # Basic progression - essential for iron crafting
    Achievement.MAKE_WOOD_PICKAXE.value: {"reward": 1.0, "enabled": True},
    
    # Stone progression - needed for better mining
    Achievement.MAKE_STONE_PICKAXE.value: {"reward": 1.0, "enabled": True},
    
    # Iron crafting goals - final objectives
    Achievement.MAKE_IRON_PICKAXE.value: {"reward": 1.0, "enabled": True},
    Achievement.MAKE_IRON_SWORD.value: {"reward": 1.0, "enabled": True},
    
    # Optional achievements (can be toggled off to find minimal set)
    Achievement.MAKE_WOOD_SWORD.value: {"reward": 1.0, "enabled": False},
    Achievement.MAKE_STONE_SWORD.value: {"reward": 1.0, "enabled": False},
}

@jax.jit
def my_ppo_harvesting_reward_fn_state(prev_state: EnvState, current_state: EnvState, done: jnp.ndarray) -> jnp.float32:
    prev_achievements = prev_state.achievements.astype(jnp.float32)
    cur_achievements = current_state.achievements.astype(jnp.float32)
    achievement_deltas = cur_achievements - prev_achievements
    
    total_reward = 0.0
    
    # Process each configured achievement
    for achievement_id, config in HARVEST_ACHIEVEMENT_REWARDS.items():
        if config["enabled"]:
            # Add reward for newly achieved accomplishments
            achievement_gained = achievement_deltas[achievement_id]
            total_reward += achievement_gained * config["reward"]
    
    # Return 0 if episode done, otherwise return calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    return jnp.array(reward, dtype=jnp.float32)

def my_ppo_crafting_reward_fn_state(prev_state: EnvState, current_state: EnvState, done: jnp.ndarray) -> jnp.float32:
    prev_achievements = prev_state.achievements.astype(jnp.float32)
    cur_achievements = current_state.achievements.astype(jnp.float32)
    achievement_deltas = cur_achievements - prev_achievements
    
    total_reward = 0.0
    
    # Process each configured achievement
    for achievement_id, config in CRAFTING_ACHIEVEMENT_REWARDS.items():
        if config["enabled"]:
            # Add reward for newly achieved accomplishments
            achievement_gained = achievement_deltas[achievement_id]
            total_reward += achievement_gained * config["reward"]
    
    # Return 0 if episode done, otherwise return calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    return jnp.array(reward, dtype=jnp.float32)

@jax.jit
def my_harvesting_crafting_reward_fn_state(prev_state: EnvState, current_state: EnvState, done: jnp.ndarray) -> jnp.float32:
    # Get harvesting reward
    # harvesting_reward = my_harvesting_reward_fn_state(prev_state, current_state, done)
    harvesting_reward = my_ppo_harvesting_reward_fn_state(prev_state, current_state, done)
    
    # Get crafting reward
    crafting_reward = my_crafting_reward_fn_state(prev_state, current_state, done)
    # crafting_reward = my_ppo_crafting_reward_fn_state(prev_state, current_state, done)
    
    # Combine the rewards
    combined_reward = harvesting_reward + crafting_reward
    
    return combined_reward.astype(jnp.float32)