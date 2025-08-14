import jax
import jax.numpy as jnp
from flax import struct

# ===== Achievements index map (for readability) =====
A_T100       = 0
A_T250       = 1
A_T500       = 2
A_T750       = 3
A_T1000      = 4
A_WOOD       = 5
A_WOOD_PX    = 6
A_STONE      = 7
A_STONE_PX   = 8
A_ATE        = 9
A_DRANK      = 10
A_SLEPT      = 11
A_NODMG200   = 12
A_ENERGY_UP  = 13

# ===== Update to number of achievements =====
NUM_CUSTOM_ACHIEVEMENTS = 14


@struct.dataclass
class CustomAchievementTracker:
    # One-time flags
    achievements: jnp.ndarray  # shape: (NUM_CUSTOM_ACHIEVEMENTS,), dtype=bool
    # Minimal state for streaks
    no_damage_streak: jnp.int32 = jnp.int32(0)


def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        no_damage_streak=jnp.int32(0),
    )


def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        # Update streaks
        took_damage = cur_state.player_health < prev_state.player_health
        new_no_damage = jnp.where(took_damage, jnp.int32(0), prev_tracker.no_damage_streak + jnp.int32(1))

        # Event predicates (scalars, jnp.bool_)
        p_time100  = cur_state.timestep >= 100
        p_time250  = cur_state.timestep >= 250
        p_time500  = cur_state.timestep >= 500
        p_time750  = cur_state.timestep >= 750
        p_time1000 = cur_state.timestep >= 1000

        p_wood     = (prev_state.inventory.wood == 0)        & (cur_state.inventory.wood > 0)
        p_wood_px  = (prev_state.inventory.wood_pickaxe == 0) & (cur_state.inventory.wood_pickaxe > 0)
        p_stone    = (prev_state.inventory.stone == 0)       & (cur_state.inventory.stone > 0)
        p_stone_px = (prev_state.inventory.stone_pickaxe == 0) & (cur_state.inventory.stone_pickaxe > 0)

        p_ate      = cur_state.player_food  > prev_state.player_food
        p_drank    = cur_state.player_drink > prev_state.player_drink
        p_slept    = (~prev_state.is_sleeping) & (cur_state.is_sleeping)
        p_nd200    = new_no_damage >= 200
        p_energy   = cur_state.player_energy > prev_state.player_energy

        # Pack predicates in index order
        preds = jnp.array([
            p_time100, p_time250, p_time500, p_time750, p_time1000,
            p_wood, p_wood_px, p_stone, p_stone_px,
            p_ate, p_drank, p_slept, p_nd200, p_energy
        ], dtype=jnp.bool_)

        # One-time set: OR keeps them latched once true
        new_ach = prev_tracker.achievements | preds

        return prev_tracker.replace(
            achievements=new_ach,
            no_damage_streak=new_no_damage
        )

    # Reset tracker cleanly if episode done
    return jax.lax.cond(done, init_single_tracker, update_achievements)


def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)
