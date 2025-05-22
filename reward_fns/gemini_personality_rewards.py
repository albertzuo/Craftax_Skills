import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import *

NUM_BLOCK_TYPES = len(BlockType)     # Number of block types (e.g., len(BlockType))
NUM_MOB_TYPES = 4        # Number of mob types (zombie, cow, skeleton, arrow)

all_map_flat_size = OBS_DIM[0] * OBS_DIM[1] * (NUM_BLOCK_TYPES + NUM_MOB_TYPES)
inventory_size = 12
intrinsics_size = 4 # health, food, drink, energy
direction_size = 4
misc_size = 2 # light_level, is_sleeping

inventory_start_idx = all_map_flat_size
intrinsics_start_idx = inventory_start_idx + inventory_size
direction_start_idx = intrinsics_start_idx + intrinsics_size
misc_start_idx = direction_start_idx + direction_size

health_idx = intrinsics_start_idx + 0
food_idx = intrinsics_start_idx + 1
drink_idx = intrinsics_start_idx + 2
energy_idx = intrinsics_start_idx + 3
light_level_idx = misc_start_idx + 0
is_sleeping_idx = misc_start_idx + 1

# Define indices for relevant raw materials within the inventory part of the observation vector.
# These correspond to the order in the `render_craftax_symbolic` function's inventory array:
# [wood, stone, coal, iron, diamond, sapling, wood_pickaxe, stone_pickaxe, ...]
WOOD_IDX = 0
STONE_IDX = 1
COAL_IDX = 2
IRON_IDX = 3
DIAMOND_IDX = 4
SAPLING_IDX = 5
WOOD_PICKAXE_IDX = 6
STONE_PICKAXE_IDX = 7
IRON_PICKAXE_IDX = 8
WOOD_SWORD_IDX = 9
STONE_SWORD_IDX = 10
IRON_SWORD_IDX = 11

# --- Reward Functions ---

def cautious_reward_function(prev_obs: jnp.ndarray, current_obs: jnp.ndarray) -> jnp.float32:
    """
    Rewards prioritizing survival, maintaining high vitals, avoiding damage,
    and seeking safe conditions.
    """
    reward = 0.0

    # 1. Maintain Vitals (Health, Food, Drink, Energy)
    # Reward for keeping vitals high, penalty for them being low.
    # Obs values are scaled by 10, so 1.0 is max.
    reward += (current_obs[health_idx] - 0.7) * 0.5  # Encourage health above 70%
    reward += (current_obs[food_idx] - 0.5) * 0.2
    reward += (current_obs[drink_idx] - 0.5) * 0.2
    reward += (current_obs[energy_idx] - 0.3) * 0.1 # Energy is less critical than direct health/sustenance

    # 2. Penalize Health Loss Severely
    health_delta = current_obs[health_idx] - prev_obs[health_idx]
    # Only penalize loss (health_delta < 0)
    reward += jnp.minimum(0.0, health_delta) * 10.0  # Strong penalty for taking damage

    # 3. Encourage Safety (Light Level & Sleeping)
    # Reward for being in well-lit areas, especially if not sleeping.
    # light_level is 0-1. Reward for higher light.
    reward += (current_obs[light_level_idx] - 0.4) * 0.5

    # Significant reward for being asleep, especially if it's dark (implying night)
    # and player wasn't sleeping before (to reward the act of going to sleep safely)
    is_currently_sleeping = current_obs[is_sleeping_idx] > 0.5
    was_previously_sleeping = prev_obs[is_sleeping_idx] > 0.5
    is_dark = current_obs[light_level_idx] < 0.3

    # Reward for starting to sleep in the dark or continuing to sleep in the dark
    went_to_sleep_in_dark = jnp.logical_and(is_currently_sleeping, jnp.logical_not(was_previously_sleeping))
    went_to_sleep_in_dark_bonus = jnp.logical_and(went_to_sleep_in_dark, is_dark)
    reward += jax.lax.select(went_to_sleep_in_dark_bonus, 2.0, 0.0)

    stayed_asleep_in_dark_bonus = jnp.logical_and(is_currently_sleeping, was_previously_sleeping)
    stayed_asleep_in_dark_bonus = jnp.logical_and(stayed_asleep_in_dark_bonus, is_dark)
    reward += jax.lax.select(stayed_asleep_in_dark_bonus, 0.1, 0.0) # Small sustain reward

    # 4. Penalize low vitals explicitly
    reward += jax.lax.select(current_obs[health_idx] < 0.3, -2.0, 0.0) # Critical health
    reward += jax.lax.select(current_obs[food_idx] < 0.2, -1.0, 0.0)   # Starving
    reward += jax.lax.select(current_obs[drink_idx] < 0.2, -1.0, 0.0) # Dehydrated

    # 5. Preparedness: Having a weapon
    has_any_sword = jnp.logical_or(current_obs[WOOD_SWORD_IDX] > 0,
                                 jnp.logical_or(current_obs[STONE_SWORD_IDX] > 0,
                                                current_obs[IRON_SWORD_IDX] > 0))
    reward += jax.lax.select(has_any_sword, 0.5, -0.5) # Reward for having a sword, penalty otherwise

    return jnp.asarray(reward, dtype=jnp.float32)


def driven_reward_function(prev_obs: jnp.ndarray, current_obs: jnp.ndarray) -> jnp.float32:
    """
    Rewards progress, resource acquisition (especially rarer ones),
    crafting better tools, and achieving milestones.
    """
    reward = 0.0

    # 1. Resource Acquisition (rewarding *increase* in quantity)
    # Weights indicate perceived value for a "driven" agent
    reward += jnp.maximum(0.0, current_obs[WOOD_IDX] - prev_obs[WOOD_IDX]) * 0.1    # Scaled by 10, so 0.1 is 1 wood
    reward += jnp.maximum(0.0, current_obs[STONE_IDX] - prev_obs[STONE_IDX]) * 0.3
    reward += jnp.maximum(0.0, current_obs[COAL_IDX] - prev_obs[COAL_IDX]) * 0.5
    reward += jnp.maximum(0.0, current_obs[IRON_IDX] - prev_obs[IRON_IDX]) * 1.0
    reward += jnp.maximum(0.0, current_obs[DIAMOND_IDX] - prev_obs[DIAMOND_IDX]) * 3.0

    # 2. Crafting Better Tools (significant one-time bonuses for new tools)
    # Wood Tier
    newly_crafted_wood_pick = (prev_obs[WOOD_PICKAXE_IDX] == 0) & (current_obs[WOOD_PICKAXE_IDX] > 0)
    reward += jax.lax.select(newly_crafted_wood_pick, 2.0, 0.0)
    newly_crafted_wood_sword = (prev_obs[WOOD_SWORD_IDX] == 0) & (current_obs[WOOD_SWORD_IDX] > 0)
    reward += jax.lax.select(newly_crafted_wood_sword, 1.5, 0.0)

    # Stone Tier
    newly_crafted_stone_pick = (prev_obs[STONE_PICKAXE_IDX] == 0) & (current_obs[STONE_PICKAXE_IDX] > 0)
    reward += jax.lax.select(newly_crafted_stone_pick, 4.0, 0.0)
    newly_crafted_stone_sword = (prev_obs[STONE_SWORD_IDX] == 0) & (current_obs[STONE_SWORD_IDX] > 0)
    reward += jax.lax.select(newly_crafted_stone_sword, 3.0, 0.0)

    # Iron Tier
    newly_crafted_iron_pick = (prev_obs[IRON_PICKAXE_IDX] == 0) & (current_obs[IRON_PICKAXE_IDX] > 0)
    reward += jax.lax.select(newly_crafted_iron_pick, 7.0, 0.0)
    newly_crafted_iron_sword = (prev_obs[IRON_SWORD_IDX] == 0) & (current_obs[IRON_SWORD_IDX] > 0)
    reward += jax.lax.select(newly_crafted_iron_sword, 6.0, 0.0)

    # 3. Penalize Health Loss (as it hinders progress, but less than Cautious)
    health_delta = current_obs[health_idx] - prev_obs[health_idx]
    reward += jnp.minimum(0.0, health_delta) * 2.0 # Moderate penalty

    # 4. Penalize Inactivity / Stagnation (e.g. low energy, not acquiring things)
    reward += jax.lax.select(current_obs[energy_idx] < 0.2, -0.5, 0.0) # Low energy hinders drive

    # Optional: Could add a small penalty if no new resources or tools were acquired for many steps,
    # but this requires state beyond prev_obs/current_obs or more complex logic.

    return jnp.asarray(reward, dtype=jnp.float32)


def playful_reward_function(prev_obs: jnp.ndarray, current_obs: jnp.ndarray) -> jnp.float32:
    """
    Rewards diverse activities, exploration, novel interactions, and using varied items.
    This is the hardest to quantify without more direct measures of novelty or interaction.
    """
    reward = 0.0

    # 1. Encourage Collection of Diverse Items (not just quantity of one type)
    # Reward for increasing the *number of types* of items possessed.
    prev_inventory = prev_obs[inventory_start_idx:intrinsics_start_idx]
    current_inventory = current_obs[inventory_start_idx:intrinsics_start_idx]

    # Count non-zero item types (items are scaled by 10.0, so >0.0 means at least 0.1, i.e. 1 item)
    prev_item_types_count = jnp.sum(prev_inventory > 1e-4) # Use a small epsilon
    current_item_types_count = jnp.sum(current_inventory > 1e-4)
    reward += jnp.maximum(0.0, current_item_types_count - prev_item_types_count) * 1.5

    # 2. Encourage using/having a variety of tools (not just the "best" one)
    # Small bonus for having each type of tool, encouraging crafting them even if not "optimal" for driven.
    # This is a state reward, not a delta.
    has_wood_pick = current_obs[WOOD_PICKAXE_IDX] > 0
    has_stone_pick = current_obs[STONE_PICKAXE_IDX] > 0
    has_iron_pick = current_obs[IRON_PICKAXE_IDX] > 0
    reward += jax.lax.select(has_wood_pick, 0.1, 0.0)
    reward += jax.lax.select(has_stone_pick, 0.1, 0.0)
    reward += jax.lax.select(has_iron_pick, 0.1, 0.0)
    # Similar for swords
    has_wood_sword = current_obs[WOOD_SWORD_IDX] > 0
    has_stone_sword = current_obs[STONE_SWORD_IDX] > 0
    has_iron_sword = current_obs[IRON_SWORD_IDX] > 0
    reward += jax.lax.select(has_wood_sword, 0.1, 0.0)
    reward += jax.lax.select(has_stone_sword, 0.1, 0.0)
    reward += jax.lax.select(has_iron_sword, 0.1, 0.0)


    # 3. Encourage Movement / Minor Exploration (change in light level as a proxy)
    light_level_delta = jnp.abs(current_obs[light_level_idx] - prev_obs[light_level_idx])
    # Reward if change is noticeable, suggesting movement between areas (e.g. cave to surface)
    reward += jax.lax.select(light_level_delta > 0.2, 0.3, 0.0)

    # 4. Maintain moderate energy for playfulness
    reward += (current_obs[energy_idx] - 0.5) * 0.2 # Encourage energy around 50% or more

    # 5. Slight penalty for health loss (playful doesn't mean reckless, but less averse than Cautious)
    health_delta = current_obs[health_idx] - prev_obs[health_idx]
    reward += jnp.minimum(0.0, health_delta) * 1.0

    # 6. Interaction with environment (very hard to specify without action details)
    # Placeholder: If we could detect "using an item" or "placing a block" that isn't strictly
    # for survival or immediate progress, that could be rewarded.
    # For now, diversity of inventory and tools is the main proxy.

    # 7. Avoid being static - small penalty if core vitals and inventory barely change.
    # This is a rough heuristic.
    obs_diff_core = jnp.sum(jnp.abs(current_obs[inventory_start_idx:direction_start_idx] - \
                                   prev_obs[inventory_start_idx:direction_start_idx]))
    reward += jax.lax.select(obs_diff_core < 0.01, -0.1, 0.0) # Penalize if inventory and vitals are static

    return jnp.asarray(reward, dtype=jnp.float32)