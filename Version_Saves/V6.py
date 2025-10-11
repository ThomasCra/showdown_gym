import os
import time
from typing import Any, Dict

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player


from showdown_gym.base_environment import BaseShowdownEnv


class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one

    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return None  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        Modified: Only allow switch actions (0-5) if active pokemon is fainted.
        Otherwise, force move actions (6-9).

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        # # Check if current pokemon is fainted
        # if self.battle1 and self.battle1.active_pokemon:
        #     current_pokemon_fainted = self.battle1.active_pokemon.fainted
            
        #     # If pokemon is NOT fainted and agent tries to switch (0-5), convert to move action
        #     if not current_pokemon_fainted and 0 <= action <= 5:
        #         # Convert switch action to move action (6-9 range)
        #         # Map switch actions 0-5 to move actions 6-9 (only 4 moves available)
        #         action = 6 + (action % 4)
            
        #     # If pokemon IS fainted and agent tries to do move actions, allow only switches
        #     elif current_pokemon_fainted and action >= 6:
        #         # Convert move action to switch action (0-5 range)
        #         action = action % 6
        
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Enhanced debugging information for RL agent performance monitoring
        if self.battle1 is not None:
            agent = self.possible_agents[0]  # Get the agent's name
            battle = self.battle1
            
            # === Basic Battle Outcome Info ===
            info[agent]["win"] = battle.won  # Store win status (True/False)
            info[agent]["battle_finished"] = battle.finished
            info[agent]["turns"] = battle.turn
            
            # === Damage and HP Tracking ===
            # Total damage dealt to opponent (cumulative)
            info[agent]["damage_dealt"] = float(
                sum([mon.max_hp - mon.current_hp for mon in battle.opponent_team.values() 
                     if mon.max_hp is not None and mon.current_hp is not None])
            )
            
            # Total damage taken by agent (cumulative)
            info[agent]["damage_taken"] = float(
                sum([mon.max_hp - mon.current_hp for mon in battle.team.values() 
                     if mon.max_hp is not None and mon.current_hp is not None])
            )
            
            # Damage ratios for performance analysis
            total_team_max_hp = sum([mon.max_hp for mon in battle.team.values() if mon.max_hp is not None])
            total_opponent_max_hp = sum([mon.max_hp for mon in battle.opponent_team.values() if mon.max_hp is not None])
            
            info[agent]["damage_dealt_ratio"] = float(info[agent]["damage_dealt"] / max(total_opponent_max_hp, 1))
            info[agent]["damage_taken_ratio"] = float(info[agent]["damage_taken"] / max(total_team_max_hp, 1))
            
            # === Pokémon Status Tracking ===
            info[agent]["pokemon_left"] = sum([not mon.fainted for mon in battle.team.values()])
            info[agent]["opponent_pokemon_left"] = sum([not mon.fainted for mon in battle.opponent_team.values()])
            info[agent]["pokemon_fainted"] = sum([mon.fainted for mon in battle.team.values()])
            info[agent]["opponent_pokemon_fainted"] = sum([mon.fainted for mon in battle.opponent_team.values()])
            
            # === Active Pokémon Info ===
            if battle.active_pokemon:
                info[agent]["active_pokemon_hp"] = float(battle.active_pokemon.current_hp_fraction or 0.0)
                info[agent]["active_pokemon_species"] = battle.active_pokemon.species
                info[agent]["active_pokemon_fainted"] = battle.active_pokemon.fainted
                info[agent]["active_pokemon_status"] = str(battle.active_pokemon.status) if battle.active_pokemon.status else "none"
                
                # Active Pokémon stats for debugging stat calculations
                if hasattr(battle.active_pokemon, 'stats') and battle.active_pokemon.stats:
                    info[agent]["active_pokemon_stats"] = {
                        "attack": battle.active_pokemon.stats.get('atk', 0),
                        "defense": battle.active_pokemon.stats.get('def', 0),
                        "sp_attack": battle.active_pokemon.stats.get('spa', 0),
                        "sp_defense": battle.active_pokemon.stats.get('spd', 0),
                        "speed": battle.active_pokemon.stats.get('spe', 0),
                        "hp": battle.active_pokemon.stats.get('hp', 0)
                    }
                
                # Boosts/debuffs for strategic analysis
                if hasattr(battle.active_pokemon, 'boosts') and battle.active_pokemon.boosts:
                    info[agent]["active_pokemon_boosts"] = dict(battle.active_pokemon.boosts)
            
            # === Opponent Active Pokémon Info ===
            if battle.opponent_active_pokemon:
                info[agent]["opponent_active_hp"] = float(battle.opponent_active_pokemon.current_hp_fraction or 0.0)
                info[agent]["opponent_active_species"] = battle.opponent_active_pokemon.species
                info[agent]["opponent_active_fainted"] = battle.opponent_active_pokemon.fainted
                info[agent]["opponent_active_status"] = str(battle.opponent_active_pokemon.status) if battle.opponent_active_pokemon.status else "none"
            
            # === Available Actions Info ===
            info[agent]["available_moves_count"] = len(battle.available_moves)
            info[agent]["available_switches_count"] = len(battle.available_switches)
            
            # Move details for debugging move selection
            if battle.available_moves:
                info[agent]["available_moves"] = []
                for move in battle.available_moves:
                    move_info = {
                        "id": move.id,
                        "base_power": move.base_power,
                        "accuracy": move.accuracy,
                        "priority": move.priority,
                        "type": str(move.type) if move.type else "none"
                    }
                    
                    # Type effectiveness calculation for debugging
                    if move.type and battle.opponent_active_pokemon:
                        try:
                            effectiveness = move.type.damage_multiplier(
                                battle.opponent_active_pokemon.type_1,
                                battle.opponent_active_pokemon.type_2,
                                type_chart=battle._format.type_chart if hasattr(battle, '_format') else None
                            )
                            move_info["type_effectiveness"] = float(effectiveness)
                        except (AttributeError, TypeError):
                            move_info["type_effectiveness"] = 1.0
                    
                    info[agent]["available_moves"].append(move_info)
            
            # === Team Composition Info ===
            team_hp_percentages = []
            team_species = []
            for mon in battle.team.values():
                team_hp_percentages.append(float(mon.current_hp_fraction or 0.0))
                team_species.append(mon.species)
            
            info[agent]["team_hp_percentages"] = team_hp_percentages
            info[agent]["team_species"] = team_species
            
            # === Battle Field Conditions ===
            info[agent]["weather"] = str(battle.weather) if hasattr(battle, 'weather') and battle.weather else "none"
            
            # Side conditions (like hazards)
            if hasattr(battle, 'side_conditions'):
                info[agent]["side_conditions"] = [str(condition) for condition in battle.side_conditions]
                info[agent]["opponent_side_conditions"] = [str(condition) for condition in battle.opponent_side_conditions] if hasattr(battle, 'opponent_side_conditions') else []
            
            # === Performance Metrics ===
            # Efficiency metrics for battle analysis
            if battle.turn > 0:
                info[agent]["damage_per_turn"] = float(info[agent]["damage_dealt"] / battle.turn)
                info[agent]["damage_taken_per_turn"] = float(info[agent]["damage_taken"] / battle.turn)
            
            # Calculate reward components for debugging reward function
            # Use try-except to handle cases where prior battle doesn't exist yet (e.g., during initial reset)
            try:
                prior_battle = self._get_prior_battle(battle)
                if prior_battle:
                    # HP differences from last turn
                    def safe_hp_sum(team_dict, attr):
                        return sum(getattr(mon, attr, 0) or 0 for mon in team_dict.values() if mon.max_hp is not None)
                    
                    current_team_hp = safe_hp_sum(battle.team, 'current_hp')
                    current_opponent_hp = safe_hp_sum(battle.opponent_team, 'current_hp')
                    prior_team_hp = safe_hp_sum(prior_battle.team, 'current_hp')
                    prior_opponent_hp = safe_hp_sum(prior_battle.opponent_team, 'current_hp')
                    
                    info[agent]["step_damage_dealt"] = float(prior_opponent_hp - current_opponent_hp)
                    info[agent]["step_damage_taken"] = float(prior_team_hp - current_team_hp)
                    
                    # Knockout tracking
                    current_fainted_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted])
                    prior_fainted_opponent = len([mon for mon in prior_battle.opponent_team.values() if mon.fainted])
                    current_fainted_team = len([mon for mon in battle.team.values() if mon.fainted])
                    prior_fainted_team = len([mon for mon in prior_battle.team.values() if mon.fainted])
                    
                    info[agent]["step_knockouts_dealt"] = current_fainted_opponent - prior_fainted_opponent
                    info[agent]["step_knockouts_taken"] = current_fainted_team - prior_fainted_team
                else:
                    # No prior battle available (initial state)
                    info[agent]["step_damage_dealt"] = 0.0
                    info[agent]["step_damage_taken"] = 0.0
                    info[agent]["step_knockouts_dealt"] = 0
                    info[agent]["step_knockouts_taken"] = 0
            except AttributeError:
                # Handle case where _prior_battle_one doesn't exist yet
                info[agent]["step_damage_dealt"] = 0.0
                info[agent]["step_damage_taken"] = 0.0
                info[agent]["step_knockouts_dealt"] = 0
                info[agent]["step_knockouts_taken"] = 0
            
            # === Error Tracking for Debugging ===
            # Track if there are any issues with the battle state
            info[agent]["debug_flags"] = {
                "has_active_pokemon": battle.active_pokemon is not None,
                "has_opponent_active": battle.opponent_active_pokemon is not None,
                "has_available_moves": len(battle.available_moves) > 0,
                "has_available_switches": len(battle.available_switches) > 0,
                "observation_size_matches": len(self.embed_battle(battle)) == self._observation_size()
            }

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Enhanced reward function that provides detailed feedback for better RL learning.
        
        Combines multiple reward components:
        - HP-based rewards for damage dealt/taken
        - Knockout rewards for fainting opponent Pokémon  
        - Victory/defeat bonuses
        - Strategic rewards for type effectiveness and switching
        - Time penalty to encourage decisive play

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
        Returns:
            float: The calculated reward based on the current battle state and changes.
        """
        
        # Get prior battle for comparison
        prior_battle = self._get_prior_battle(battle)
        if prior_battle is None:
            # If no prior battle, return base reward from helper
            return self.reward_computing_helper(
                battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
            )
        
        reward = 0.0
        
        # === 1. HP-based rewards (damage dealt vs taken) ===
        def safe_hp_sum(team_dict, attr):
            return sum(getattr(mon, attr, 0) or 0 for mon in team_dict.values() if mon.max_hp is not None)
        
        # Current HP states
        current_team_hp = safe_hp_sum(battle.team, 'current_hp')
        current_opponent_hp = safe_hp_sum(battle.opponent_team, 'current_hp')
        prior_team_hp = safe_hp_sum(prior_battle.team, 'current_hp')
        prior_opponent_hp = safe_hp_sum(prior_battle.opponent_team, 'current_hp')
        
        # Reward for damage dealt to opponent, penalty for damage taken
        damage_dealt = prior_opponent_hp - current_opponent_hp
        damage_taken = prior_team_hp - current_team_hp
        
        reward += damage_dealt * 0.01  # Reward for dealing damage
        reward -= damage_taken * 0.015  # Penalty for taking damage (slightly higher)
        
        # === 2. Knockout rewards (fainting opponent Pokémon) ===
        current_fainted_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted])
        prior_fainted_opponent = len([mon for mon in prior_battle.opponent_team.values() if mon.fainted])
        current_fainted_team = len([mon for mon in battle.team.values() if mon.fainted])
        prior_fainted_team = len([mon for mon in prior_battle.team.values() if mon.fainted])
        
        # Big rewards for knocking out opponent Pokémon
        knockouts_dealt = current_fainted_opponent - prior_fainted_opponent
        knockouts_taken = current_fainted_team - prior_fainted_team
        
        reward += knockouts_dealt * 5.0   # Large reward for knocking out opponent
        reward -= knockouts_taken * 6.0   # Large penalty for losing own Pokémon
        
        # === 3. Victory/Defeat rewards ===
        if battle.finished:
            if battle.won:
                reward += 50.0  # Large victory bonus
                # Additional bonus for winning with Pokémon remaining
                remaining_pokemon = 6 - current_fainted_team
                reward += remaining_pokemon * 2.0
            else:
                reward -= 30.0  # Defeat penalty
        
        # === 4. Strategic play rewards ===
        
        # Type effectiveness bonus (encourage smart move selection)
        if battle.active_pokemon and battle.opponent_active_pokemon and len(battle.available_moves) > 0:
            # Check if the last move used was super effective
            try:
                for move in battle.available_moves[:1]:  # Check first available move as proxy
                    if move.type and hasattr(move.type, 'damage_multiplier'):
                        effectiveness = move.type.damage_multiplier(
                            battle.opponent_active_pokemon.type_1,
                            battle.opponent_active_pokemon.type_2,
                            type_chart=battle._format.type_chart if hasattr(battle, '_format') else None
                        )
                        if effectiveness > 1.0:
                            reward += 0.1  # Small bonus for having super effective moves available
                        elif effectiveness < 1.0:
                            reward -= 0.05  # Small penalty for not very effective moves
            except (AttributeError, TypeError):
                pass
        
        # Switching strategy reward (encourage strategic switching when at low HP)
        if battle.active_pokemon and prior_battle.active_pokemon:
            if (battle.active_pokemon.species != prior_battle.active_pokemon.species and 
                prior_battle.active_pokemon.current_hp_fraction and 
                prior_battle.active_pokemon.current_hp_fraction < 0.25):
                reward += 1.0  # Reward for strategic switching when low on HP
        
        # === 5. Efficiency incentives ===
        
        # Small time penalty to encourage decisive play (avoid stalling)
        reward -= 0.02  # Small penalty per turn to encourage efficient battles
        
        # Penalty for using ineffective moves repeatedly
        if damage_dealt == 0 and not knockouts_dealt:
            reward -= 0.1  # Penalty for turns without progress
        
        # === 6. Status and field effect considerations ===
        
        # Small rewards for applying status conditions
        if battle.active_pokemon and prior_battle.active_pokemon:
            if (battle.opponent_active_pokemon and prior_battle.opponent_active_pokemon and
                battle.opponent_active_pokemon.status and not prior_battle.opponent_active_pokemon.status):
                reward += 0.5  # Reward for inflicting status
        
        # Normalize reward to reasonable range
        reward = np.clip(reward, -20.0, 20.0)
        
        return float(reward)

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Updated to match the enhanced embed_battle method
        # 4 (move base power) + 4 (move dmg multiplier) + 4 (move accuracy) + 4 (move priority) +
        # 2 (active pokemon hp) + 4 (active pokemon stats) + 2 (active pokemon status) +
        # 2 (fainted counts) + 6 (team hp) + 6 (opponent team hp) + 1 (weather) + 1 (turn) = 40
        return 40

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        Enhanced version with more comprehensive battle state information.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        # Move information (4 moves max)
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        moves_accuracy = np.ones(4)
        moves_priority = np.zeros(4)
        
        for i, move in enumerate(battle.available_moves):
            # Base power (normalized to 0-1.5 range)
            moves_base_power[i] = move.base_power / 100 if move.base_power else 0
            
            # Type effectiveness
            if move.type and battle.opponent_active_pokemon:
                try:
                    moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=battle._format.type_chart if hasattr(battle, '_format') else None
                    )
                except (AttributeError, TypeError):
                    moves_dmg_multiplier[i] = 1.0
            
            # Accuracy (normalized to 0-1 range)
            moves_accuracy[i] = move.accuracy / 100 if move.accuracy else 1.0
            
            # Priority (normalized to -1 to 1 range) - safely handle missing priority
            try:
                moves_priority[i] = move.priority / 5.0 if move.priority else 0.0
            except (KeyError, AttributeError):
                moves_priority[i] = 0.0  # Default priority of 0

        # Active Pokémon information
        active_hp = 0.0
        opponent_active_hp = 0.0
        
        if battle.active_pokemon and not battle.active_pokemon.fainted:
            active_hp = battle.active_pokemon.current_hp_fraction
        
        if battle.opponent_active_pokemon and not battle.opponent_active_pokemon.fainted:
            opponent_active_hp = battle.opponent_active_pokemon.current_hp_fraction

        # Active Pokémon stats (normalized by typical stat ranges)
        active_stats = np.zeros(4)  # [atk, def, spa, spd]
        if battle.active_pokemon and not battle.active_pokemon.fainted:
            mon = battle.active_pokemon
            # Normalize stats (typical range is 50-200 for base stats)
            active_stats[0] = (mon.stats.get('atk', 100) - 50) / 150  # Attack
            active_stats[1] = (mon.stats.get('def', 100) - 50) / 150  # Defense  
            active_stats[2] = (mon.stats.get('spa', 100) - 50) / 150  # Special Attack
            active_stats[3] = (mon.stats.get('spd', 100) - 50) / 150  # Special Defense

        # Status conditions (binary encoding)
        active_status = np.zeros(2)  # [has_status, is_boosted]
        if battle.active_pokemon and not battle.active_pokemon.fainted:
            active_status[0] = 1.0 if battle.active_pokemon.status else 0.0
            # Check if pokemon has stat boosts
            boosts = battle.active_pokemon.boosts
            active_status[1] = 1.0 if any(boosts.values()) else 0.0

        # Team information
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        
        # Team HP percentages (for strategic switching decisions)
        team_hp = np.zeros(6)
        opponent_team_hp = np.zeros(6)
        
        team_mons = list(battle.team.values())
        for i in range(min(6, len(team_mons))):
            if team_mons[i].current_hp_fraction is not None:
                team_hp[i] = team_mons[i].current_hp_fraction
        
        opponent_mons = list(battle.opponent_team.values())
        for i in range(min(6, len(opponent_mons))):
            if opponent_mons[i].current_hp_fraction is not None:
                opponent_team_hp[i] = opponent_mons[i].current_hp_fraction

        # Weather and field conditions (simplified)
        weather = 0.0
        if hasattr(battle, 'weather') and battle.weather:
            # Simple binary encoding for weather presence
            weather = 1.0
        
        # Turn number (normalized)
        turn_normalized = min(battle.turn / 50.0, 1.0)  # Cap at 50 turns

        #########################################################################################################
        # Enhanced final vector with 40 components for better learning
        #########################################################################################################
        
        final_vector = np.concatenate([
            moves_base_power,      # 4 features
            moves_dmg_multiplier,  # 4 features  
            moves_accuracy,        # 4 features
            moves_priority,        # 4 features
            [active_hp, opponent_active_hp],  # 2 features
            active_stats,          # 4 features
            active_status,         # 2 features
            [fainted_mon_team, fainted_mon_opponent],  # 2 features
            team_hp,              # 6 features
            opponent_team_hp,     # 6 features
            [weather],            # 1 feature
            [turn_normalized],    # 1 feature
        ])
        
        return np.float32(final_vector)


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None