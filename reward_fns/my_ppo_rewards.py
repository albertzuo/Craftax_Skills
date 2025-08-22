import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import *
from craftax.craftax_classic.constants import Achievement

# Configurable achievement rewards for iron crafting progression
# Official Craftax gives 1.0 reward per achievement, so we maintain that
ACHIEVEMENT_REWARDS = {
    # Basic progression - essential for iron crafting
    Achievement.COLLECT_WOOD.value: {"reward": 1.0, "enabled": False},
    Achievement.PLACE_TABLE.value: {"reward": 1.0, "enabled": False},
    Achievement.MAKE_WOOD_PICKAXE.value: {"reward": 1.0, "enabled": True},
    
    # Stone progression - needed for better mining
    Achievement.COLLECT_STONE.value: {"reward": 1.0, "enabled": False},
    Achievement.MAKE_STONE_PICKAXE.value: {"reward": 1.0, "enabled": True},
    
    # Iron prerequisites - furnace needed for smelting
    Achievement.PLACE_FURNACE.value: {"reward": 1.0, "enabled": False},
    Achievement.COLLECT_COAL.value: {"reward": 1.0, "enabled": False},
    Achievement.COLLECT_IRON.value: {"reward": 1.0, "enabled": True},
    
    # Iron crafting goals - final objectives
    Achievement.MAKE_IRON_PICKAXE.value: {"reward": 1.0, "enabled": True},
    Achievement.MAKE_IRON_SWORD.value: {"reward": 1.0, "enabled": False},
    
    # Optional achievements (can be toggled off to find minimal set)
    Achievement.COLLECT_SAPLING.value: {"reward": 1.0, "enabled": False},
    Achievement.COLLECT_DRINK.value: {"reward": 1.0, "enabled": False},
    Achievement.EAT_COW.value: {"reward": 1.0, "enabled": False},
    Achievement.EAT_PLANT.value: {"reward": 1.0, "enabled": False},
    Achievement.DEFEAT_ZOMBIE.value: {"reward": 1.0, "enabled": False},
    Achievement.DEFEAT_SKELETON.value: {"reward": 1.0, "enabled": False},
    Achievement.WAKE_UP.value: {"reward": 1.0, "enabled": False},
    Achievement.COLLECT_DIAMOND.value: {"reward": 1.0, "enabled": False},
    Achievement.PLACE_STONE.value: {"reward": 1.0, "enabled": False},
    Achievement.PLACE_PLANT.value: {"reward": 1.0, "enabled": False},
    Achievement.MAKE_WOOD_SWORD.value: {"reward": 1.0, "enabled": False},
    Achievement.MAKE_STONE_SWORD.value: {"reward": 1.0, "enabled": False},
}

@jax.jit
def configurable_achievement_reward_fn(prev_state, cur_state, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates reward based on configurable achievement progression.
    Focuses on the minimal set needed to learn iron crafting.
    """
    # Calculate achievement differences
    prev_achievements = prev_state.achievements.astype(jnp.float32)
    cur_achievements = cur_state.achievements.astype(jnp.float32)
    achievement_deltas = cur_achievements - prev_achievements
    
    total_reward = 0.0
    
    # Process each configured achievement
    for achievement_id, config in ACHIEVEMENT_REWARDS.items():
        if config["enabled"]:
            # Add reward for newly achieved accomplishments
            achievement_gained = achievement_deltas[achievement_id]
            total_reward += achievement_gained * config["reward"]
    
    # Return 0 if episode done, otherwise return calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    return jnp.array(reward, dtype=jnp.float32)

def no_iron_configurable_achievement_reward_fn(prev_state, cur_state, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates reward based on configurable achievement progression.
    Focuses on the minimal set needed to learn iron crafting.
    """
    # Calculate achievement differences
    prev_achievements = prev_state.achievements.astype(jnp.float32)
    cur_achievements = cur_state.achievements.astype(jnp.float32)
    achievement_deltas = cur_achievements - prev_achievements
    
    total_reward = 0.0
    
    NO_IRON_ACHIEVEMENT_REWARDS = {
        Achievement.MAKE_WOOD_PICKAXE.value: {"reward": 1.0, "enabled": True},
        Achievement.MAKE_STONE_PICKAXE.value: {"reward": 1.0, "enabled": True},
        Achievement.COLLECT_IRON.value: {"reward": 1.0, "enabled": False},
        Achievement.COLLECT_COAL.value: {"reward": 1.0, "enabled": True},        
        Achievement.MAKE_IRON_PICKAXE.value: {"reward": 1.0, "enabled": True},
    }
    # Process each configured achievement
    for achievement_id, config in NO_IRON_ACHIEVEMENT_REWARDS.items():
        if config["enabled"]:
            # Add reward for newly achieved accomplishments
            achievement_gained = achievement_deltas[achievement_id]
            total_reward += achievement_gained * config["reward"]
    
    # Return 0 if episode done, otherwise return calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    return jnp.array(reward, dtype=jnp.float32)

def no_stone_pick_configurable_achievement_reward_fn(prev_state, cur_state, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates reward based on configurable achievement progression.
    Focuses on the minimal set needed to learn iron crafting.
    """
    # Calculate achievement differences
    prev_achievements = prev_state.achievements.astype(jnp.float32)
    cur_achievements = cur_state.achievements.astype(jnp.float32)
    achievement_deltas = cur_achievements - prev_achievements
    
    total_reward = 0.0
    
    NO_IRON_ACHIEVEMENT_REWARDS = {
        Achievement.MAKE_WOOD_PICKAXE.value: {"reward": 1.0, "enabled": True},
        Achievement.MAKE_STONE_PICKAXE.value: {"reward": 1.0, "enabled": False},
        Achievement.COLLECT_IRON.value: {"reward": 1.0, "enabled": True},
        Achievement.COLLECT_COAL.value: {"reward": 1.0, "enabled": True},        
        Achievement.MAKE_IRON_PICKAXE.value: {"reward": 1.0, "enabled": True},
    }
    # Process each configured achievement
    for achievement_id, config in NO_IRON_ACHIEVEMENT_REWARDS.items():
        if config["enabled"]:
            # Add reward for newly achieved accomplishments
            achievement_gained = achievement_deltas[achievement_id]
            total_reward += achievement_gained * config["reward"]
    
    # Return 0 if episode done, otherwise return calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    return jnp.array(reward, dtype=jnp.float32)

def no_iron_pick_configurable_achievement_reward_fn(prev_state, cur_state, done: jnp.ndarray) -> jnp.float32:
    """
    Calculates reward based on configurable achievement progression.
    Focuses on the minimal set needed to learn iron crafting.
    """
    # Calculate achievement differences
    prev_achievements = prev_state.achievements.astype(jnp.float32)
    cur_achievements = cur_state.achievements.astype(jnp.float32)
    achievement_deltas = cur_achievements - prev_achievements
    
    total_reward = 0.0
    
    NO_IRON_ACHIEVEMENT_REWARDS = {
        Achievement.MAKE_WOOD_PICKAXE.value: {"reward": 1.0, "enabled": True},
        Achievement.MAKE_STONE_PICKAXE.value: {"reward": 1.0, "enabled": True},
        Achievement.COLLECT_IRON.value: {"reward": 1.0, "enabled": True},
        Achievement.COLLECT_COAL.value: {"reward": 1.0, "enabled": True},        
        Achievement.MAKE_IRON_PICKAXE.value: {"reward": 1.0, "enabled": False},
    }
    # Process each configured achievement
    for achievement_id, config in NO_IRON_ACHIEVEMENT_REWARDS.items():
        if config["enabled"]:
            # Add reward for newly achieved accomplishments
            achievement_gained = achievement_deltas[achievement_id]
            total_reward += achievement_gained * config["reward"]
    
    # Return 0 if episode done, otherwise return calculated reward
    reward = jnp.where(done, 0.0, total_reward)
    return jnp.array(reward, dtype=jnp.float32)