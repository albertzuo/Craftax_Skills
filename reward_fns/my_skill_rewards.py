import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.constants import Achievement

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

HARVESTING_REWARD_WEIGHTS = jnp.array([
    0.5,  # Wood
    2.0,  # Stone
    5.0,  # Coal
    10.0,  # Iron
    20.0, # Diamond
    0.1,  # Sapling
], dtype=jnp.float32)
HARVESTING_MULTIPLIER = 0.5

CRAFTED_ITEM_INDICES = jnp.array([
    6,  # wood_pickaxe
    7,  # stone_pickaxe
    8,  # iron_pickaxe
    9,  # wood_sword
    10, # stone_sword
    11, # iron_sword
])
CRAFTING_MULTIPLIER = 2.5

# Weights applied to the reward for crafting each corresponding item.
# Higher weights encourage crafting more advanced items.
# Order must match CRAFTED_ITEM_INDICES:
# [wood_p, stone_p, iron_p, wood_s, stone_s, iron_s]
CRAFTING_REWARD_WEIGHTS = jnp.array([
    2.0,  # Reward for wood_pickaxe
    10.0,  # Reward for stone_pickaxe
    20.0,  # Reward for iron_pickaxe
    2.0,  # Reward for wood_sword
    10.0,  # Reward for stone_sword
    20.0,  # Reward for iron_sword
])

def get_inventory_slice(obs: jnp.ndarray) -> jnp.ndarray:
    """
    Extracts the inventory part from the flattened observation vector.

    Assumes the inventory is a fixed-size slice located before the final
    'intrinsics', 'direction', 'light_level', and 'is_sleeping' elements.

    Args:
        obs: The flattened observation array for a single time step.

    Returns:
        The inventory slice of the observation array.
    """
    # The observation structure concatenates:
    # [flat_map, inventory, intrinsics, direction, light_level, is_sleeping]
    # Sizes:
    # inventory: 12 (wood, stone, coal, iron, diamond, sapling, 3 pickaxes, 3 swords)
    # intrinsics: 4 (health, food, drink, energy)
    # direction: 4 (one-hot encoded)
    # light_level: 1
    # is_sleeping: 1
    # Total elements after inventory = 4 + 4 + 1 + 1 = 10
    inventory_size = 12
    num_trailing_elements = 10

    # Calculate start and end indices dynamically based on the total length
    # Works correctly even if the map size changes, as long as the
    # inventory and trailing elements remain constant.
    start_idx = obs.shape[-1] - num_trailing_elements - inventory_size
    end_idx = obs.shape[-1] - num_trailing_elements

    # Ensure indices are valid (useful for debugging)
    # assert start_idx >= 0, f"Calculated start index {start_idx} is invalid."
    # assert end_idx <= obs.shape[-1], f"Calculated end index {end_idx} is invalid."
    # assert (end_idx - start_idx) == inventory_size, f"Slice size {(end_idx - start_idx)} != {inventory_size}"

    return obs[..., start_idx:end_idx]

@jax.jit
def my_harvesting_reward_fn(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, done: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates a reward signal focused on harvesting raw materials in Craftax.

    The reward is based on the positive change in the quantity of specific raw
    materials (wood, stone, coal, iron, diamond, sapling) in the agent's
    inventory between the previous and current observation.

    Args:
        prev_obs: The flattened observation array from the previous time step.
        current_obs: The flattened observation array from the current time step.

    Returns:
        A scalar JAX array representing the reward for the current step.
    """
    # Extract inventory slices from both observations
    # Note: The observation stores inventory counts divided by 10.0,
    # so we multiply by 10.0 to get the actual counts.
    prev_inventory = get_inventory_slice(prev_obs) * 10.0
    current_inventory = get_inventory_slice(current_obs) * 10.0

    # Select only the raw materials we care about for this skill
    # prev_raw_materials = jnp.minimum(prev_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)], 1) # Select first 6 items
    # current_raw_materials = jnp.minimum(current_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)], 1) # Select first 6 items

    # Calculate the change in counts for each raw material
    delta_materials = current_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)] - prev_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)]#current_raw_materials - prev_raw_materials

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
def my_crafting_reward_fn(prev_obs_flat: jnp.ndarray, current_obs_flat: jnp.ndarray, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates a reward signal focused on the crafting skill in Craftax.

    This function rewards the agent based on the *increase* in the quantity
    of specific crafted items (tools and swords) in its inventory between
    two consecutive timesteps.

    Args:
        prev_obs_flat: The flattened observation vector from the previous timestep.
                       Expected shape: (total_flat_size,)
        current_obs_flat: The flattened observation vector from the current timestep.
                          Expected shape: (total_flat_size,)

    Returns:
        A scalar float32 reward value. Positive reward is given when a target
        item count increases, weighted by CRAFTING_REWARD_WEIGHTS.
    """
    # --- 1. Extract Inventory Data ---
    # Use jax.lax.dynamic_slice_in_dim for efficient slicing on accelerators.
    # Extracts the segment corresponding to the inventory from the flat observation vectors.
    prev_inventory_scaled = get_inventory_slice(prev_obs_flat) # CHANGED FROM DEFAULT LOGIC FROM GEMINI
    current_inventory_scaled = get_inventory_slice(current_obs_flat)

    
    center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2   
    crafting_table_idx = BlockType.CRAFTING_TABLE.value
    furnace_idx = BlockType.FURNACE.value
    
    map_obs = current_obs_flat[:all_map_flat_size]
    map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    local_area = map_obs[center_x-1:center_x+2, center_y-1:center_y+2, :NUM_BLOCK_TYPES]    
    is_near_crafting_table = jnp.any(local_area[:, :, crafting_table_idx] > 0)
    is_near_furnace = jnp.any(local_area[:, :, furnace_idx] > 0)
    is_in_iron_crafting_pos = jnp.logical_and(is_near_crafting_table, is_near_furnace)

    prev_map_obs = prev_obs_flat[:all_map_flat_size]
    prev_map_obs = prev_map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    prev_local_area = prev_map_obs[center_x-1:center_x+2, center_y-1:center_y+2, :NUM_BLOCK_TYPES]
    prev_is_near_crafting_table = jnp.any(prev_local_area[:, :, crafting_table_idx] > 0)
    prev_is_near_furnace = jnp.any(prev_local_area[:, :, furnace_idx] > 0)
    prev_is_in_iron_crafting_pos = jnp.logical_and(prev_is_near_crafting_table, prev_is_near_furnace)

    # Reward for entering crafting position
    entered_crafting_pos_reward = jnp.where(
        jnp.logical_and(jnp.logical_not(prev_is_in_iron_crafting_pos), is_in_iron_crafting_pos),
        0.5,
        0.0
    )

    entered_crafting_pos_reward += jnp.where(
        jnp.logical_and(prev_is_in_iron_crafting_pos, jnp.logical_not(is_in_iron_crafting_pos)),
        -0.5,
        0.0
    )

    # --- 2. Convert to Item Counts ---
    # The inventory values in the observation are scaled (divided by 10.0).
    # Multiply by 10.0 and round to nearest integer to get actual counts.
    # Cast to int32 for count representation.
    prev_inventory_counts = jnp.round(prev_inventory_scaled * 10.0).astype(jnp.int32)
    current_inventory_counts = jnp.round(current_inventory_scaled * 10.0).astype(jnp.int32)

    # --- 3. Isolate Target Crafted Item Counts ---
    # Select only the counts of the items we want to reward (pickaxes, swords)
    # using the predefined indices.
    # should 
    prev_crafted_item_counts = jnp.minimum(prev_inventory_counts[CRAFTED_ITEM_INDICES], 1)
    current_crafted_item_counts = jnp.minimum(current_inventory_counts[CRAFTED_ITEM_INDICES], 1)  # Cap at 1

    # --- 4. Calculate Increase in Counts ---
    # Compute the difference between current and previous counts for target items.
    delta_counts = current_crafted_item_counts - prev_crafted_item_counts

    # --- 5. Apply Weights and Sum ---A
    # Multiply the increase in count for each item by its corresponding weight.
    weighted_increase = delta_counts * CRAFTING_REWARD_WEIGHTS * CRAFTING_MULTIPLIER

    # Sum the weighted increases across all target items to get the final reward.
    total_reward = jnp.sum(weighted_increase) #+ entered_crafting_pos_reward * CRAFTING_MULTIPLIER

    # Return 0 if done, otherwise return the calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    
    # --- 6. Return Reward ---
    # Ensure the reward is a float32, a common type for RL rewards.
    return reward.astype(jnp.float32)

@jax.jit
def my_survival_reward_fn(prev_obs, obs, done):
    """
    Calculates a reward signal focused on survival in the Craftax environment.

    Args:
        obs: The flattened observation vector for the current state
             (output of render_craftax_symbolic).
        prev_obs: The flattened observation vector for the previous state.

    Returns:
        A scalar reward value (float).
    """
    reward = 0.0

    # --- Extract Player Intrinsics (Current State) ---
    # Rescale from [0, 1] back to original values (e.g., 0-10)
    health = round(obs[health_idx] * 10.0)
    food = round(obs[food_idx] * 10.0)
    drink = round(obs[drink_idx] * 10.0)
    energy = round(obs[energy_idx] * 10.0)
    is_sleeping = round(obs[is_sleeping_idx]) # Already 0 or 1

    # --- Extract Previous Health ---
    prev_health = round(prev_obs[health_idx] * 10.0)
    prev_food = round(prev_obs[food_idx] * 10.0)
    prev_drink = round(prev_obs[drink_idx] * 10.0)
    prev_energy = round(prev_obs[energy_idx] * 10.0)
    intrinsic_stat_multiplier = 0.2

    # --- Mob Proximity Penalty ---
    center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    
    # Extract map observation and reshape to spatial format
    map_obs = obs[:all_map_flat_size]
    map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    
    # Get mob layer (last NUM_MOB_TYPES channels)
    mob_layer = map_obs[:, :, NUM_BLOCK_TYPES:]
    
    # Define hostile mob indices (zombie=0, skeleton=2 based on common Craftax setup)
    zombie_idx = 0
    skeleton_idx = 2
    
    # Check 3x3 area around player for hostile mobs
    local_area = mob_layer[center_x-1:center_x+2, center_y-1:center_y+2, :]
    
    # Penalty for being near hostile mobs
    nearby_zombies = jnp.sum(local_area[:, :, zombie_idx])
    nearby_skeletons = jnp.sum(local_area[:, :, skeleton_idx])
    
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
    reward = jnp.where(done, -10.0, reward)

    return reward


### -----------------------

@jax.jit
def my_harvesting_reward_fn_v2(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, done: jnp.ndarray, prev_state, cur_state) -> jnp.ndarray:
    """
    Calculates a reward signal focused on harvesting raw materials in Craftax.

    The reward is based on the positive change in the quantity of specific raw
    materials (wood, stone, coal, iron, diamond, sapling) in the agent's
    inventory between the previous and current observation.

    Args:
        prev_obs: The flattened observation array from the previous time step.
        current_obs: The flattened observation array from the current time step.

    Returns:
        A scalar JAX array representing the reward for the current step.
    """
    # Extract inventory slices from both observations
    # Note: The observation stores inventory counts divided by 10.0,
    # so we multiply by 10.0 to get the actual counts.
    prev_inventory = get_inventory_slice(prev_obs) * 10.0
    current_inventory = get_inventory_slice(current_obs) * 10.0

    # Select only the raw materials we care about for this skill
    prev_raw_materials = jnp.minimum(prev_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)], 1) # Select first 6 items
    current_raw_materials = jnp.minimum(current_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)], 1) # Select first 6 items

    # Calculate the change in counts for each raw material
    delta_materials = current_raw_materials - prev_raw_materials

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
def my_crafting_reward_fn_v2(prev_obs_flat: jnp.ndarray, current_obs_flat: jnp.ndarray, done: jnp.ndarray, prev_state, cur_state) -> jnp.float32:
    """
    Calculates a reward signal focused on the crafting skill in Craftax.

    This function rewards the agent based on the *increase* in the quantity
    of specific crafted items (tools and swords) in its inventory between
    two consecutive timesteps.

    Args:
        prev_obs_flat: The flattened observation vector from the previous timestep.
                       Expected shape: (total_flat_size,)
        current_obs_flat: The flattened observation vector from the current timestep.
                          Expected shape: (total_flat_size,)

    Returns:
        A scalar float32 reward value. Positive reward is given when a target
        item count increases, weighted by CRAFTING_REWARD_WEIGHTS.
    """
    # --- 1. Extract Inventory Data ---
    # Use jax.lax.dynamic_slice_in_dim for efficient slicing on accelerators.
    # Extracts the segment corresponding to the inventory from the flat observation vectors.
    prev_inventory_scaled = get_inventory_slice(prev_obs_flat) # CHANGED FROM DEFAULT LOGIC FROM GEMINI
    current_inventory_scaled = get_inventory_slice(current_obs_flat)

    
    center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2   
    crafting_table_idx = BlockType.CRAFTING_TABLE.value
    furnace_idx = BlockType.FURNACE.value
    
    map_obs = current_obs_flat[:all_map_flat_size]
    map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    local_area = map_obs[center_x-1:center_x+2, center_y-1:center_y+2, :NUM_BLOCK_TYPES]    
    is_near_crafting_table = jnp.any(local_area[:, :, crafting_table_idx] > 0)
    is_near_furnace = jnp.any(local_area[:, :, furnace_idx] > 0)
    is_in_iron_crafting_pos = jnp.logical_and(is_near_crafting_table, is_near_furnace)

    prev_map_obs = prev_obs_flat[:all_map_flat_size]
    prev_map_obs = prev_map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    prev_local_area = prev_map_obs[center_x-1:center_x+2, center_y-1:center_y+2, :NUM_BLOCK_TYPES]
    prev_is_near_crafting_table = jnp.any(prev_local_area[:, :, crafting_table_idx] > 0)
    prev_is_near_furnace = jnp.any(prev_local_area[:, :, furnace_idx] > 0)
    prev_is_in_iron_crafting_pos = jnp.logical_and(prev_is_near_crafting_table, prev_is_near_furnace)

    # Reward for entering crafting position
    entered_crafting_pos_reward = jnp.where(
        jnp.logical_and(jnp.logical_not(prev_is_in_iron_crafting_pos), is_in_iron_crafting_pos),
        0.5,
        0.0
    )

    entered_crafting_pos_reward += jnp.where(
        jnp.logical_and(prev_is_in_iron_crafting_pos, jnp.logical_not(is_in_iron_crafting_pos)),
        -0.5,
        0.0
    )

    # --- 2. Convert to Item Counts ---
    # The inventory values in the observation are scaled (divided by 10.0).
    # Multiply by 10.0 and round to nearest integer to get actual counts.
    # Cast to int32 for count representation.
    prev_inventory_counts = jnp.round(prev_inventory_scaled * 10.0).astype(jnp.int32)
    current_inventory_counts = jnp.round(current_inventory_scaled * 10.0).astype(jnp.int32)

    # --- 3. Isolate Target Crafted Item Counts ---
    # Select only the counts of the items we want to reward (pickaxes, swords)
    # using the predefined indices.
    # should 
    prev_crafted_item_counts = jnp.minimum(prev_inventory_counts[CRAFTED_ITEM_INDICES], 1)
    current_crafted_item_counts = jnp.minimum(current_inventory_counts[CRAFTED_ITEM_INDICES], 1)  # Cap at 1

    # --- 4. Calculate Increase in Counts ---
    # Compute the difference between current and previous counts for target items.
    delta_counts = current_crafted_item_counts - prev_crafted_item_counts

    # --- 5. Apply Weights and Sum ---A
    # Multiply the increase in count for each item by its corresponding weight.
    weighted_increase = delta_counts * CRAFTING_REWARD_WEIGHTS * CRAFTING_MULTIPLIER

    # Reward for placing furnace (achievement difference check)
    place_furnace_reward = jnp.where(
        jnp.logical_and(
            prev_state.achievements[Achievement.PLACE_FURNACE.value] == 0,
            cur_state.achievements[Achievement.PLACE_FURNACE.value] == 1
        ),
        1.0,
        0.0
    )

    # Sum the weighted increases across all target items to get the final reward.
    total_reward = jnp.sum(weighted_increase) + place_furnace_reward #+ entered_crafting_pos_reward * CRAFTING_MULTIPLIER

    # Return 0 if done, otherwise return the calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    
    # --- 6. Return Reward ---
    # Ensure the reward is a float32, a common type for RL rewards.
    return reward.astype(jnp.float32)

@jax.jit
def my_survival_reward_fn_v2(prev_obs, obs, done, prev_state, cur_state):
    """
    Calculates a reward signal focused on survival in the Craftax environment.

    Args:
        obs: The flattened observation vector for the current state
             (output of render_craftax_symbolic).
        prev_obs: The flattened observation vector for the previous state.

    Returns:
        A scalar reward value (float).
    """
    reward = 0.0

    # --- Extract Player Intrinsics (Current State) ---
    # Rescale from [0, 1] back to original values (e.g., 0-10)
    health = round(obs[health_idx] * 10.0)
    food = round(obs[food_idx] * 10.0)
    drink = round(obs[drink_idx] * 10.0)
    energy = round(obs[energy_idx] * 10.0)
    is_sleeping = round(obs[is_sleeping_idx]) # Already 0 or 1

    # --- Extract Previous Health ---
    prev_health = round(prev_obs[health_idx] * 10.0)
    prev_food = round(prev_obs[food_idx] * 10.0)
    prev_drink = round(prev_obs[drink_idx] * 10.0)
    prev_energy = round(prev_obs[energy_idx] * 10.0)
    intrinsic_stat_multiplier = 0.2

    # --- Mob Proximity Penalty ---
    center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    
    # Extract map observation and reshape to spatial format
    map_obs = obs[:all_map_flat_size]
    map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    
    # Get mob layer (last NUM_MOB_TYPES channels)
    mob_layer = map_obs[:, :, NUM_BLOCK_TYPES:]
    
    # Define hostile mob indices (zombie=0, skeleton=2 based on common Craftax setup)
    zombie_idx = 0
    skeleton_idx = 2
    
    # Check 3x3 area around player for hostile mobs
    local_area = mob_layer[center_x-1:center_x+2, center_y-1:center_y+2, :]
    
    # Penalty for being near hostile mobs
    nearby_zombies = jnp.sum(local_area[:, :, zombie_idx])
    nearby_skeletons = jnp.sum(local_area[:, :, skeleton_idx])
    
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
    reward = jnp.where(done, -10.0, reward)

    return reward