import jax
import jax.numpy as jnp
from flax import struct

# Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 8

@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # Shape: (NUM_CUSTOM_ACHIEVEMENTS,)
    # intermediate variables here if needed

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_)
    )

# Achievement 0: First Wood
def update_achievement_first_wood(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.wood > 0,
        prev_state.inventory.wood == 0,
    )
    
    new_achievements = prev_tracker.achievements.at[0].set(
        jnp.logical_or(prev_tracker.achievements[0], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 1: Wooden Pickaxe
def update_achievement_wooden_pickaxe(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.wood_pickaxe > 0,
        prev_state.inventory.wood_pickaxe == 0  
    )
    
    new_achievements = prev_tracker.achievements.at[1].set(
        jnp.logical_or(prev_tracker.achievements[1], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 2: First Stone
def update_achievement_first_stone(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.stone > 0,
        prev_state.inventory.stone == 0
    )
    
    new_achievements = prev_tracker.achievements.at[2].set(
        jnp.logical_or(prev_tracker.achievements[2], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 3: Stone Pickaxe
def update_achievement_stone_pickaxe(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.stone_pickaxe > 0,
        prev_state.inventory.stone_pickaxe == 0
    )
    
    new_achievements = prev_tracker.achievements.at[3].set(
        jnp.logical_or(prev_tracker.achievements[3], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 4: First Coal
def update_achievement_first_coal(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.coal > 0,
        prev_state.inventory.coal == 0
    )
    
    new_achievements = prev_tracker.achievements.at[4].set(
        jnp.logical_or(prev_tracker.achievements[4], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 5: First Iron
def update_achievement_first_iron(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.iron > 0,
        prev_state.inventory.iron == 0
    )

    new_achievements = prev_tracker.achievements.at[5].set(
        jnp.logical_or(prev_tracker.achievements[5], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 6: Iron Pickaxe
def update_achievement_iron_pickaxe(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.iron_pickaxe > 0,
        prev_state.inventory.iron_pickaxe == 0
    )

    new_achievements = prev_tracker.achievements.at[6].set(
        jnp.logical_or(prev_tracker.achievements[6], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

# Achievement 7: Diamond (Main Goal)
def update_achievement_diamond(prev_state, cur_state, prev_tracker):
    achievement_unlocked = jnp.logical_and(
        cur_state.inventory.diamond > 0,
        prev_state.inventory.diamond == 0
    )

    new_achievements = prev_tracker.achievements.at[7].set(
        jnp.logical_or(prev_tracker.achievements[7], achievement_unlocked)
    )
    
    return prev_tracker.replace(achievements=new_achievements)

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tracker = prev_tracker        
        tracker = update_achievement_first_wood(prev_state, cur_state, tracker)
        tracker = update_achievement_wooden_pickaxe(prev_state, cur_state, tracker)
        tracker = update_achievement_first_stone(prev_state, cur_state, tracker)
        tracker = update_achievement_stone_pickaxe(prev_state, cur_state, tracker)
        tracker = update_achievement_first_coal(prev_state, cur_state, tracker)
        tracker = update_achievement_first_iron(prev_state, cur_state, tracker)
        tracker = update_achievement_iron_pickaxe(prev_state, cur_state, tracker)
        tracker = update_achievement_diamond(prev_state, cur_state, tracker)
        return tracker
    
    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)
