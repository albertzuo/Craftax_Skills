import jax
import jax.numpy as jnp
from flax import struct

# Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 25

@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # (NUM_CUSTOM_ACHIEVEMENTS,)
    # Track exploration from spawn (surface world, no caves)
    spawn_pos: jnp.ndarray     # shape (2,), int32
    max_radius: jnp.int32      # L1 distance from spawn (max so far)

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        spawn_pos=jnp.array([-1, -1], dtype=jnp.int32),
        max_radius=jnp.int32(0),
    )

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    # Thresholds
    HIGH_LEVEL = 6
    FAST_TIME_LIMIT = 3000  # timely diamond bonus within episode (max ~10k)

    def update_achievements():
        tracker = prev_tracker
        inv = cur_state.inventory

        # Initialize spawn on first step
        new_spawn = jax.lax.cond(
            tracker.spawn_pos[0] < 0,
            lambda: cur_state.player_position,
            lambda: tracker.spawn_pos,
        )
        tracker = tracker.replace(spawn_pos=new_spawn)

        # Update exploration radius (surface-only exploration proxy)
        cur_radius = jnp.sum(jnp.abs(cur_state.player_position - tracker.spawn_pos)).astype(jnp.int32)
        new_max_radius = jnp.maximum(tracker.max_radius, cur_radius)
        tracker = tracker.replace(max_radius=new_max_radius)

        # Convenience
        swords_total = (inv.wood_sword + inv.stone_sword + inv.iron_sword)
        fell_asleep = jnp.logical_and(prev_state.is_sleeping == False,
                                      cur_state.is_sleeping == True)

        # === 25 Achievements ===
        conds = [
            # A) Early resources (6)
            inv.wood >= 1,                                   # 0: First wood
            inv.wood >= 4,                                   # 1: Wood stockpile
            inv.wood_pickaxe >= 1,                           # 2: Craft wooden pickaxe
            inv.stone >= 1,                                  # 3: First stone
            inv.stone >= 8,                                  # 4: Stone stockpile
            inv.coal >= 1,                                   # 5: First coal

            # B) Tool progression to iron (4)
            inv.stone_pickaxe >= 1,                          # 6: Craft stone pickaxe
            inv.iron >= 1,                                   # 7: First iron
            inv.iron >= 3,                                   # 8: Iron stockpile
            inv.iron_pickaxe >= 1,                           # 9: Craft iron pickaxe

            # C) Surface exploration (4) – replace cave/dark cues
            tracker.max_radius >= 5,                         # 10: Leave spawn area
            tracker.max_radius >= 10,                        # 11: Explore medium radius
            tracker.max_radius >= 15,                        # 12: Explore wide radius
            jnp.logical_and(inv.iron_pickaxe >= 1,
                            tracker.max_radius >= 12),       # 13: Explore widely AFTER iron pickaxe

            # D) Survival readiness (5)
            cur_state.player_food  >= HIGH_LEVEL,            # 14: Adequate food
            cur_state.player_drink >= HIGH_LEVEL,            # 15: Adequate drink
            fell_asleep,                                     # 16: Sleep once
            cur_state.player_energy >= HIGH_LEVEL,           # 17: Adequate energy
            jnp.logical_and(jnp.logical_and(cur_state.player_food  >= HIGH_LEVEL,
                                            cur_state.player_drink >= HIGH_LEVEL),
                             cur_state.player_energy >= HIGH_LEVEL),     # 18: All healthy together

            # E) Tighten iron path + speed (2)
            jnp.logical_and(inv.stone_pickaxe >= 1, inv.iron >= 1),      # 19: Get iron *with* stone pickaxe
            jnp.logical_and(inv.diamond >= 1, cur_state.timestep <= FAST_TIME_LIMIT),  # 20: Timely diamond

            # F) Diamond & efficiency (4)
            inv.diamond >= 1,                                # 21: First diamond (main goal)
            inv.diamond >= 2,                                # 22: Second diamond (optional but aligned)
            jnp.logical_and(inv.iron_pickaxe >= 1, swords_total == 0),   # 23: Iron pickaxe with no swords
            jnp.logical_and(inv.diamond >= 1, swords_total == 0),        # 24: Diamond with no swords
        ]

        # Sticky (one-way) achievements
        new_flags = jnp.array(conds, dtype=jnp.bool_)
        tracker = tracker.replace(achievements=jnp.logical_or(tracker.achievements, new_flags))
        return tracker

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)
