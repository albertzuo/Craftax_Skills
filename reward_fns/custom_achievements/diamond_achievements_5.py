import jax
import jax.numpy as jnp
from flax import struct

# 1) Update the count
NUM_CUSTOM_ACHIEVEMENTS = 5

# Indices for readability
ACH_WOOD_PICKAXE   = 0
ACH_STONE_PICKAXE  = 1
ACH_GAIN_IRON      = 2
ACH_IRON_PICKAXE   = 3
ACH_GET_DIAMOND    = 4

@struct.dataclass
class CustomAchievementTracker:
    # Unlock-once flags (per episode)
    achievements: jnp.ndarray  # shape: (NUM_CUSTOM_ACHIEVEMENTS,), dtype=bool

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_)
    )

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tracker = prev_tracker
        ach = tracker.achievements

        # Event-style triggers: fire only when the underlying quantity increases.
        wood_px_crafted  = (cur_state.inventory.wood_pickaxe  > prev_state.inventory.wood_pickaxe)
        stone_px_crafted = (cur_state.inventory.stone_pickaxe > prev_state.inventory.stone_pickaxe)
        iron_gained      = (cur_state.inventory.iron          > prev_state.inventory.iron)
        iron_px_crafted  = (cur_state.inventory.iron_pickaxe  > prev_state.inventory.iron_pickaxe)
        diamond_gained   = (cur_state.inventory.diamond       > prev_state.inventory.diamond)

        # Set-and-hold booleans (unlock once)
        ach = ach.at[ACH_WOOD_PICKAXE].set(  ach[ACH_WOOD_PICKAXE]  | wood_px_crafted)
        ach = ach.at[ACH_STONE_PICKAXE].set( ach[ACH_STONE_PICKAXE] | stone_px_crafted)
        ach = ach.at[ACH_GAIN_IRON].set(     ach[ACH_GAIN_IRON]     | iron_gained)
        ach = ach.at[ACH_IRON_PICKAXE].set(  ach[ACH_IRON_PICKAXE]  | iron_px_crafted)
        ach = ach.at[ACH_GET_DIAMOND].set(   ach[ACH_GET_DIAMOND]   | diamond_gained)

        return CustomAchievementTracker(achievements=ach)

    # Reset tracker cleanly at episode end
    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)
