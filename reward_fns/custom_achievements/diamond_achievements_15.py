import jax
import jax.numpy as jnp
from flax import struct

# Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 15

@struct.dataclass
class CustomAchievementTracker:
    # Indices map (for reference):
    #  0 First wood
    #  1 Wood stockpile (>=4)
    #  2 Craft wooden pickaxe
    #  3 First stone mined while having wood pickaxe
    #  4 Stone stockpile (>=8)
    #  5 Craft stone pickaxe
    #  6 First coal
    #  7 Coal stockpile (>=2)
    #  8 First iron mined while having stone pickaxe
    #  9 Iron stockpile for tools (>=3)
    # 10 Craft iron pickaxe
    # 11 Enter low light (cave-ready scouting)
    # 12 Low light while carrying iron pickaxe
    # 13 Preparedness: food>=3 and drink>=3
    # 14 First diamond obtained
    achievements: jnp.ndarray  # Shape: (NUM_CUSTOM_ACHIEVEMENTS,)

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_)
    )

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        prev_inv = prev_state.inventory
        cur_inv = cur_state.inventory

        # Deltas (strict "first time increase" events)
        d_wood   = (cur_inv.wood   > prev_inv.wood)
        d_stone  = (cur_inv.stone  > prev_inv.stone)
        d_coal   = (cur_inv.coal   > prev_inv.coal)
        d_iron   = (cur_inv.iron   > prev_inv.iron)
        d_diam   = (cur_inv.diamond > prev_inv.diamond)

        d_wpick  = (cur_inv.wood_pickaxe  > prev_inv.wood_pickaxe)
        d_spick  = (cur_inv.stone_pickaxe > prev_inv.stone_pickaxe)
        d_ipick  = (cur_inv.iron_pickaxe  > prev_inv.iron_pickaxe)

        # Convenience reads
        have_wpick = (cur_inv.wood_pickaxe  > 0)
        have_spick = (cur_inv.stone_pickaxe > 0)
        have_ipick = (cur_inv.iron_pickaxe  > 0)

        low_light  = (cur_state.light_level < 0.35)
        prepared   = (cur_state.player_food >= 3) & (cur_state.player_drink >= 3)

        # Threshold helpers
        wood4  = (cur_inv.wood  >= 4)
        stone8 = (cur_inv.stone >= 8)
        coal2  = (cur_inv.coal  >= 2)
        iron3  = (cur_inv.iron  >= 3)

        # 15 achievement triggers (booleans)
        triggers = jnp.array([
            d_wood,                                # 0 First wood
            wood4,                                 # 1 Wood stockpile
            d_wpick,                               # 2 Craft wooden pickaxe
            d_stone & have_wpick,                  # 3 First stone while having wood pickaxe
            stone8,                                # 4 Stone stockpile
            d_spick,                               # 5 Craft stone pickaxe
            d_coal,                                # 6 First coal
            coal2,                                 # 7 Coal stockpile
            d_iron & have_spick,                   # 8 First iron while having stone pickaxe
            iron3,                                 # 9 Iron stockpile for tools
            d_ipick,                               # 10 Craft iron pickaxe
            low_light,                             # 11 Enter low light (likely caves/night)
            low_light & have_ipick,                # 12 Low light while carrying iron pickaxe
            prepared,                              # 13 Preparedness (food & drink)
            d_diam                                 # 14 First diamond obtained
        ], dtype=jnp.bool_)

        new_ach = jnp.logical_or(prev_tracker.achievements, triggers)
        return CustomAchievementTracker(achievements=new_ach)

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)
