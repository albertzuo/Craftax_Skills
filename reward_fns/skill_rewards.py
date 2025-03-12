import jax
import jax.numpy as jnp


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
    map_size = observation.shape[0] - 22
    inventory = observation[map_size : map_size + 12]
    # Sum over the crafted items (indices 6 to 11).
    crafting_reward = jnp.sum(inventory[6:12])
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
