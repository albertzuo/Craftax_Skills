import jax
import jax.numpy as jnp
from flax import struct


# Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 15


@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # Shape: (NUM_CUSTOM_ACHIEVEMENTS,)
    # Intermediate variables for tracking progress
    max_timestep_reached: jnp.int32
    consecutive_full_stats: jnp.int32
    has_shelter_materials: jnp.bool_
    nights_survived: jnp.int32
    peak_inventory_value: jnp.int32


def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        max_timestep_reached=jnp.int32(0),
        consecutive_full_stats=jnp.int32(0),
        has_shelter_materials=jnp.bool_(False),
        nights_survived=jnp.int32(0),
        peak_inventory_value=jnp.int32(0)
    )


def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tracker = prev_tracker
        achievements = tracker.achievements.copy()
        
        # Get current timestep
        timestep = cur_state.timestep
        
        # Calculate day/night status (assuming day_length=300 from EnvParams)
        day_length = 300
        is_night = (timestep % day_length) >= (day_length // 2)
        was_night = (prev_state.timestep % day_length) >= (day_length // 2)
        
        # Track max timestep reached
        max_timestep = jnp.maximum(tracker.max_timestep_reached, timestep)
        
        # === EARLY GAME ACHIEVEMENTS (Rewards: 0.5-1.0) ===
        
        # Achievement 0: First Steps (Survive 50 timesteps) - 0.5 reward
        achievements = achievements.at[0].set(
            achievements[0] | (timestep >= 50)
        )
        
        # Achievement 1: Early Survivor (Survive 100 timesteps) - 0.7 reward
        achievements = achievements.at[1].set(
            achievements[1] | (timestep >= 100)
        )
        
        # Achievement 2: Basic Sustenance (Have all stats > 5) - 0.5 reward
        all_stats_good = (
            (cur_state.player_health > 5) &
            (cur_state.player_food > 5) &
            (cur_state.player_drink > 5) &
            (cur_state.player_energy > 5)
        )
        achievements = achievements.at[2].set(
            achievements[2] | all_stats_good
        )
        
        # Achievement 3: Resource Gatherer (Collect wood and stone) - 0.6 reward
        has_basic_resources = (
            (cur_state.inventory.wood > 0) &
            (cur_state.inventory.stone > 0)
        )
        achievements = achievements.at[3].set(
            achievements[3] | has_basic_resources
        )
        
        # === MID GAME ACHIEVEMENTS (Rewards: 1.0-2.0) ===
        
        # Achievement 4: Toolsmith (Craft any pickaxe) - 1.0 reward
        has_pickaxe = (
            (cur_state.inventory.wood_pickaxe > 0) |
            (cur_state.inventory.stone_pickaxe > 0) |
            (cur_state.inventory.iron_pickaxe > 0)
        )
        achievements = achievements.at[4].set(
            achievements[4] | has_pickaxe
        )
        
        # Achievement 5: Armed (Craft any sword) - 1.0 reward
        has_sword = (
            (cur_state.inventory.wood_sword > 0) |
            (cur_state.inventory.stone_sword > 0) |
            (cur_state.inventory.iron_sword > 0)
        )
        achievements = achievements.at[5].set(
            achievements[5] | has_sword
        )
        
        # Achievement 6: Night Survivor (Survive a full night) - 1.5 reward
        # Track transition from night to day
        night_to_day = was_night & (~is_night)
        nights_survived = jnp.where(
            night_to_day,
            tracker.nights_survived + 1,
            tracker.nights_survived
        )
        achievements = achievements.at[6].set(
            achievements[6] | (nights_survived > 0)
        )
        
        # Achievement 7: Shelter Builder (Have materials for shelter) - 1.2 reward
        has_shelter_materials = (
            (cur_state.inventory.wood >= 5) &
            (cur_state.inventory.stone >= 3)
        )
        achievements = achievements.at[7].set(
            achievements[7] | has_shelter_materials
        )
        
        # Achievement 8: Quarter Way (Survive 250 timesteps) - 2.0 reward
        achievements = achievements.at[8].set(
            achievements[8] | (timestep >= 250)
        )
        
        # === LATE GAME ACHIEVEMENTS (Rewards: 2.0-5.0) ===
        
        # Achievement 9: Halfway There (Survive 500 timesteps) - 3.0 reward
        achievements = achievements.at[9].set(
            achievements[9] | (timestep >= 500)
        )
        
        # Achievement 10: Thriving (All stats > 7 for 10 consecutive steps) - 2.5 reward
        stats_high = (
            (cur_state.player_health >= 7) &
            (cur_state.player_food >= 7) &
            (cur_state.player_drink >= 7) &
            (cur_state.player_energy >= 7)
        )
        consecutive_full = jnp.where(
            stats_high,
            tracker.consecutive_full_stats + 1,
            jnp.int32(0)
        )
        achievements = achievements.at[10].set(
            achievements[10] | (consecutive_full >= 10)
        )
        
        # Achievement 11: Resource Rich (Have valuable inventory) - 2.0 reward
        inventory_value = (
            cur_state.inventory.wood * 1 +
            cur_state.inventory.stone * 2 +
            cur_state.inventory.coal * 3 +
            cur_state.inventory.iron * 4 +
            cur_state.inventory.diamond * 10
        )
        peak_inventory = jnp.maximum(tracker.peak_inventory_value, inventory_value)
        achievements = achievements.at[11].set(
            achievements[11] | (inventory_value >= 20)
        )
        
        # Achievement 12: Three Quarters (Survive 750 timesteps) - 4.0 reward
        achievements = achievements.at[12].set(
            achievements[12] | (timestep >= 750)
        )
        
        # Achievement 13: Near Victory (Survive 900 timesteps) - 5.0 reward
        achievements = achievements.at[13].set(
            achievements[13] | (timestep >= 900)
        )
        
        # === ULTIMATE ACHIEVEMENT ===
        
        # Achievement 14: Ultimate Survivor (Survive 1000 timesteps) - 10.0 reward
        achievements = achievements.at[14].set(
            achievements[14] | (timestep >= 1000)
        )
        
        # Update tracker with new values
        tracker = CustomAchievementTracker(
            achievements=achievements,
            max_timestep_reached=max_timestep,
            consecutive_full_stats=consecutive_full,
            has_shelter_materials=has_shelter_materials,
            nights_survived=nights_survived,
            peak_inventory_value=peak_inventory
        )
        
        return tracker
    
    return jax.lax.cond(done, init_single_tracker, update_achievements)


def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    
    # Define reward values for each achievement
    reward_values = jnp.array([
        0.5,   # Achievement 0: First Steps
        0.7,   # Achievement 1: Early Survivor
        0.5,   # Achievement 2: Basic Sustenance
        0.6,   # Achievement 3: Resource Gatherer
        1.0,   # Achievement 4: Toolsmith
        1.0,   # Achievement 5: Armed
        1.5,   # Achievement 6: Night Survivor
        1.2,   # Achievement 7: Shelter Builder
        2.0,   # Achievement 8: Quarter Way
        3.0,   # Achievement 9: Halfway There
        2.5,   # Achievement 10: Thriving
        2.0,   # Achievement 11: Resource Rich
        4.0,   # Achievement 12: Three Quarters
        5.0,   # Achievement 13: Near Victory
        10.0,  # Achievement 14: Ultimate Survivor
    ], dtype=jnp.float32)
    
    # Calculate total reward
    total_reward = jnp.sum(achievement_deltas * reward_values)
    
    # Add small continuous rewards for progress
    timestep_progress_reward = (current_tracker.max_timestep_reached - prev_tracker.max_timestep_reached) * 0.001
    
    return total_reward + timestep_progress_reward