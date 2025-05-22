import jax
import jax.numpy as jnp

# ================================================================================================
# ================================================================================================
# THESE ARE FOR CRAFTAX CLASSIC
# ================================================================================================
# ================================================================================================

# ================================================================================================
# ================================================================================================
# GPT-4 Reward Functions
# ================================================================================================
# ================================================================================================

@jax.jit
def reward_harvest_resources(observation):
    """
    Computes a reward for harvesting resources.

    The inventory is located at indices [map_size : map_size+12], where the first 6 items
    represent raw resources (wood, stone, coal, iron, diamond, sapling).

    Parameters:
      observation (jnp.array): The flattened observation.

    Returns:
      resource_reward (jnp.array): Scalar reward for resources gathered.
    """
    # Determine the length of the all_map flatten part.
    map_size = observation.shape[0] - 22
    # Extract inventory from the observation.
    inventory = observation[map_size : map_size + 12]
    # Sum over the first 6 indices (raw resources).
    resource_reward = jnp.sum(inventory[:6])
    return resource_reward


@jax.jit
def reward_craft_useful_items(observation):
    """
    Computes a reward for crafting useful items.

    The crafted items are assumed to be the last 6 entries of the inventory.

    Parameters:
      observation (jnp.array): The flattened observation.

    Returns:
      crafting_reward (jnp.array): Scalar reward for crafted items.
    """
    map_size = observation.shape[0] - 22 # map_size = 8268-22 = 8246
    inventory = observation[map_size : map_size + 12] # observation[8246:8258]
    # Sum over the crafted items (indices 6 to 11).
    crafting_reward = jnp.sum(inventory[6:12]) # observation[8252:8258]
    return crafting_reward


@jax.jit
def reward_explore_efficiently(observation):
    """
    Computes a reward for efficient exploration.

    It uses the 'all_map' part of the observation (i.e. all entries before the last 22)
    and sums their values, assuming that higher values represent newly discovered terrain.

    Parameters:
      observation (jnp.array): The flattened observation.

    Returns:
      exploration_reward (jnp.array): Scalar reward for exploration.
    """
    # Determine the map part length.
    map_size = observation.shape[0] - 22
    # Extract the map data.
    all_map_flat = observation[:map_size]
    exploration_reward = jnp.sum(all_map_flat)
    return exploration_reward

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# Claude 3.7 Sonnet Reward Functions
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
def get_vmap_compatible_reward_fn():
    """
    Returns a pure function that can be used with jax.vmap
    
    Returns:
        vmapped_reward_fn: A function compatible with jax.vmap
        init_state: A function to initialize state for parallel environments
    """
    # Convert the stateful function to a pure function for JAX
    def pure_reward_fn(obs, prev_inventory, first_step):
        # Extract inventory
        map_size = OBS_DIM[0] * OBS_DIM[1] * (len(BlockType) + 4)
        current_inventory = obs[map_size:map_size+9] * 10
        
        # Base reward
        reward = jnp.where(first_step, -0.005, 0.0)
        
        # Skip reward calculation on first step
        def calc_rewards(args):
            prev_inv, curr_inv = args
            
            # Resource weights
            resource_weights = jnp.array([1.0, 2.0, 3.0, 5.0, 10.0, 0.5, 0.0, 0.0, 0.0])
            
            # Collection rewards
            collected = jnp.maximum(0, curr_inv - prev_inv)
            resource_reward = jnp.sum(collected * resource_weights)
            
            # Tool creation rewards
            wood_pick_created = (prev_inv[6] == 0) & (curr_inv[6] > 0)
            stone_pick_created = (prev_inv[7] == 0) & (curr_inv[7] > 0)
            iron_pick_created = (prev_inv[8] == 0) & (curr_inv[8] > 0)
            
            tool_reward = wood_pick_created * 3.0 + stone_pick_created * 5.0 + iron_pick_created * 8.0
            
            # Collection rate bonus
            collection_rate = jnp.sum(jnp.maximum(0, curr_inv[:5] - prev_inv[:5]))
            rate_reward = collection_rate * 0.2
            
            return -0.005 + resource_reward + tool_reward + rate_reward
        
        # Only calculate rewards if not first step
        reward = jnp.where(
            first_step,
            -0.005,
            calc_rewards((prev_inventory, current_inventory))
        )
        
        return reward, current_inventory, False
    
    # Make this function compatible with vmap
    @jax.jit
    def vmapped_fn(observations, prev_inventories, first_steps):
        rewards, new_inventories, new_first_steps = jax.vmap(pure_reward_fn)(
            observations, prev_inventories, first_steps
        )
        return rewards, new_inventories, new_first_steps
    
    # Initialize the state for each parallel environment
    def init_state(num_envs):
        prev_inventories = jnp.zeros((num_envs, 9), dtype=jnp.float32)
        first_steps = jnp.ones((num_envs,), dtype=jnp.bool_)
        return prev_inventories, first_steps
    
    return vmapped_fn, init_state

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# GPT-4 (Take 2) Reward Functions
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

TARGET_RESOURCE_INDICES = {
    'wood': 0,
    'stone': 1,
    'coal': 2,
}

def harvester_reward(obs: jnp.ndarray, prev_obs: jnp.ndarray) -> jnp.ndarray:
    # Inventory is 12 elements starting after flattened map view
    # All inventory items are normalized to [0, 1] with /10.0
    inventory_start = - (12 + 4 + 4 + 2)  # inventory + intrinsics + direction + [light_level, is_sleeping]
    inventory_end = inventory_start + 12
    inventory = obs[inventory_start:inventory_end]
    prev_inventory = prev_obs[inventory_start:inventory_end]
    
    # Calculate resource gains
    resource_deltas = inventory - prev_inventory
    wood_gain = resource_deltas[TARGET_RESOURCE_INDICES['wood']]
    stone_gain = resource_deltas[TARGET_RESOURCE_INDICES['stone']]
    coal_gain = resource_deltas[TARGET_RESOURCE_INDICES['coal']]

    # Reward structure (scalable)
    reward = 10.0 * wood_gain + 8.0 * stone_gain + 12.0 * coal_gain

    # Survival incentive: avoid zero health
    health_idx = inventory_end
    health = obs[health_idx]
    reward += jnp.where(health <= 0.01, -20.0, 0.0)  # big penalty for dying

    # Bonus: small reward for using energy for harvesting, if energy is dropping reasonably
    energy_idx = health_idx + 3  # health, food, drink, energy
    energy = obs[energy_idx]
    prev_energy = prev_obs[energy_idx]
    energy_drop = prev_energy - energy
    reward += jnp.clip(energy_drop, 0.0, 0.05) * 2.0  # reward controlled effort

    return reward


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# GPT-4 (Take 3) Reward Functions
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def craft_item_reward(prev_obs: jnp.ndarray, curr_obs: jnp.ndarray, goal_item_idx: int) -> jnp.ndarray:
    """
    Computes a shaped reward for crafting a specific item.

    Args:
        prev_obs: Previous flattened symbolic observation.
        curr_obs: Current flattened symbolic observation.
        goal_item_idx: Index in the inventory vector corresponding to the desired crafted item.

    Returns:
        Scalar reward.
    """
    INV_START = -18  # start of the inventory vector from end of obs
    inventory_prev = prev_obs[INV_START:INV_START + 12] * 10  # unnormalize
    inventory_curr = curr_obs[INV_START:INV_START + 12] * 10

    crafted_delta = inventory_curr[goal_item_idx] - inventory_prev[goal_item_idx]

    # Success reward
    crafted_reward = jnp.where(crafted_delta > 0, 10.0, 0.0)

    # Optional shaping: reward gathering required materials
    # Example: for wooden pickaxe, index 6, require wood (0) and stone (1)
    # Define material requirements per goal
    material_requirements = {
        6: [0],          # wood pickaxe: wood
        7: [0, 1],       # stone pickaxe: wood + stone
        8: [0, 1, 3],    # iron pickaxe: wood + stone + iron
        9: [0],          # wood sword: wood
        10: [0, 1],      # stone sword: wood + stone
        11: [0, 3],      # iron sword: wood + iron
    }

    required_material_idxs = material_requirements.get(goal_item_idx, [])
    material_progress = sum(
        jnp.clip(inventory_curr[i] - inventory_prev[i], 0, 1)
        for i in required_material_idxs
    )

    shaped_reward = material_progress * 0.5  # modest shaping

    return crafted_reward + shaped_reward

def reward_mine_advanced_ores(prev_obs: jnp.ndarray, obs: jnp.ndarray) -> jnp.float32:
    """
    Reward for mining new coal or iron ores, based on inventory change.
    
    Reward:
    - +1 per unit of new coal or iron mined
    - +2 bonus if both are mined in the same step
    - Reward is only active if agent has stone or iron pickaxe
    """
    # Extract inventory (normalized between 0 and 1, max 10 units)
    obs_coal = obs[-22 + 2] * 10.0
    obs_iron = obs[-22 + 3] * 10.0
    prev_coal = prev_obs[-22 + 2] * 10.0
    prev_iron = prev_obs[-22 + 3] * 10.0

    # Tools
    has_stone_pickaxe = obs[-22 + 7] * 10.0 >= 1
    has_iron_pickaxe = obs[-22 + 8] * 10.0 >= 1
    has_valid_pickaxe = jnp.logical_or(has_stone_pickaxe, has_iron_pickaxe)

    # Delta inventory (positive gains only)
    delta_coal = jnp.maximum(obs_coal - prev_coal, 0.0)
    delta_iron = jnp.maximum(obs_iron - prev_iron, 0.0)

    base_reward = delta_coal + delta_iron  # +1 per unit
    bonus = 2.0 * jnp.logical_and(delta_coal > 0, delta_iron > 0)  # bonus for mining both
    reward = jnp.where(has_valid_pickaxe, base_reward + bonus, 0.0)

    return reward.astype(jnp.float32)



