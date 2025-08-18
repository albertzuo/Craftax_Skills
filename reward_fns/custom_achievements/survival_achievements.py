import jax
import jax.numpy as jnp
from flax import struct

# Achievements: time(8) + craft(6) + eat(3) + drink(3) + sleep(3) + nodmg(3) + heal(2) = 28
NUM_CUSTOM_ACHIEVEMENTS = 28

@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # (NUM_CUSTOM_ACHIEVEMENTS,), bool
    no_damage_streak: jnp.int32 = jnp.int32(0)
    eat_count: jnp.int32 = jnp.int32(0)
    drink_count: jnp.int32 = jnp.int32(0)
    sleep_count: jnp.int32 = jnp.int32(0)
    heal_count: jnp.int32 = jnp.int32(0)

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        no_damage_streak=jnp.int32(0),
        eat_count=jnp.int32(0),
        drink_count=jnp.int32(0),
        sleep_count=jnp.int32(0),
        heal_count=jnp.int32(0),
    )

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        # --- Events ---
        took_damage = cur_state.player_health < prev_state.player_health
        healed      = cur_state.player_health > prev_state.player_health
        ate         = cur_state.player_food   > prev_state.player_food
        drank       = cur_state.player_drink  > prev_state.player_drink
        slept_tr    = (~prev_state.is_sleeping) & (cur_state.is_sleeping)

        # --- Counters / streaks ---
        no_dmg = jnp.where(took_damage, jnp.int32(0), prev_tracker.no_damage_streak + jnp.int32(1))
        eat_c  = prev_tracker.eat_count   + ate.astype(jnp.int32)
        drink_c= prev_tracker.drink_count + drank.astype(jnp.int32)
        sleep_c= prev_tracker.sleep_count + slept_tr.astype(jnp.int32)
        heal_c = prev_tracker.heal_count  + healed.astype(jnp.int32)

        # --- Time ladder ---
        t = cur_state.timestep
        time_preds = jnp.array([
            t >= 100, t >= 250, t >= 500, t >= 600,
            t >= 700, t >= 800, t >= 900, t >= 1000
        ], dtype=jnp.bool_)

        # --- Craft / progression (first-time only) ---
        craft_preds = jnp.array([
            (prev_state.inventory.wood == 0)          & (cur_state.inventory.wood > 0),
            (prev_state.inventory.wood_pickaxe == 0)  & (cur_state.inventory.wood_pickaxe > 0),
            (prev_state.inventory.stone == 0)         & (cur_state.inventory.stone > 0),
            (prev_state.inventory.stone_pickaxe == 0) & (cur_state.inventory.stone_pickaxe > 0),
            (prev_state.inventory.iron_pickaxe == 0)  & (cur_state.inventory.iron_pickaxe > 0),
            (prev_state.inventory.stone_sword == 0)   & (cur_state.inventory.stone_sword > 0),
        ], dtype=jnp.bool_)

        # --- Sustained maintenance (cumulative counts) ---
        eat_preds   = jnp.array([eat_c >= 3,  eat_c >= 6,  eat_c >= 9 ], dtype=jnp.bool_)
        drink_preds = jnp.array([drink_c >= 3,drink_c >= 6,drink_c >= 9], dtype=jnp.bool_)
        sleep_preds = jnp.array([sleep_c >= 1,sleep_c >= 3,sleep_c >= 5], dtype=jnp.bool_)

        # --- Safety streaks & recovery ---
        nodmg_preds = jnp.array([no_dmg >= 100, no_dmg >= 200, no_dmg >= 300], dtype=jnp.bool_)
        heal_preds  = jnp.array([heal_c >= 3, heal_c >= 8], dtype=jnp.bool_)

        preds = jnp.concatenate([time_preds, craft_preds, eat_preds, drink_preds, sleep_preds, nodmg_preds, heal_preds])
        new_ach = prev_tracker.achievements | preds

        return prev_tracker.replace(
            achievements=new_ach,
            no_damage_streak=no_dmg,
            eat_count=eat_c,
            drink_count=drink_c,
            sleep_count=sleep_c,
            heal_count=heal_c,
        )

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)
