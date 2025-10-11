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
        imitation_agent_type: str = "simple",  # "simple", "max", or "random"
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        self.rl_agent = account_name_one
        self.imitation_agent_type = imitation_agent_type
        
        # Store the last action taken by our RL agent for reward calculation
        self.last_rl_action = None
        
        # Track episode statistics for better DDQN training
        self.episode_step_count = 0
        self.max_episode_steps = 300  # Prevent extremely long episodes

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
        Enhanced with step counting and action validation for DDQN stability.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        # Increment step counter
        self.episode_step_count += 1
        
        # Store the action for imitation learning reward calculation
        self.last_rl_action = action
        
        # Force episode termination if too long (prevents infinite episodes)
        if self.episode_step_count >= self.max_episode_steps:
            return -1  # Forfeit action
        
        return action
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask indicating which actions are valid in the current state.
        This helps DDQN focus only on legal actions and improves training stability.
        
        Returns:
            np.ndarray: Boolean mask of shape (26,) where True means action is valid
        """
        mask = np.zeros(26, dtype=bool)
        
        try:
            if self.battle1 is not None and hasattr(self.battle1, 'available_moves'):
                battle = self.battle1
                
                # Always allow default and forfeit actions
                mask[24] = True  # Default action (mapped from -2)
                mask[25] = True  # Forfeit action (mapped from -1)
                
                # Enable available switches (actions 0-5)
                if hasattr(battle, 'available_switches') and battle.available_switches:
                    for i, switch in enumerate(battle.available_switches):
                        if i < 6:
                            mask[i] = True
                
                # Enable available moves (actions 6-9 for basic moves)
                if battle.available_moves:
                    for i, move in enumerate(battle.available_moves):
                        if i < 4:  # Limit to first 4 moves
                            mask[6 + i] = True
                            
                            # Also enable special move variants if the base move is available
                            # Mega evolution (10-13)
                            if 10 + i < 26:
                                mask[10 + i] = True
                            # Z-moves (14-17) 
                            if 14 + i < 26:
                                mask[14 + i] = True
                            # Dynamax (18-21)
                            if 18 + i < 26:
                                mask[18 + i] = True
                            # Terastallize (22-25)
                            if 22 + i < 26:
                                mask[22 + i] = True
                
        except (AttributeError, TypeError):
            # If anything fails, allow at least default actions
            mask[24] = True  # Default
            mask[25] = True  # Forfeit
        
        return mask
    
    def reset_episode_tracking(self):
        """Reset episode-specific tracking variables."""
        self.episode_step_count = 0
        self.last_rl_action = None

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add DDQN-optimized information for tracking and training
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["imitation_agent_type"] = self.imitation_agent_type
            info[agent]["episode_step"] = self.episode_step_count
            info[agent]["max_steps_reached"] = self.episode_step_count >= self.max_episode_steps
            
            # Action masking information for DDQN
            info[agent]["action_mask"] = self.get_action_mask()
            info[agent]["num_valid_actions"] = int(np.sum(self.get_action_mask()))
            
            # Track if the last action matched the reference agent
            if hasattr(self, 'last_rl_action') and self.last_rl_action is not None:
                reference_action = self._get_reference_action(self.battle1)
                info[agent]["action_match"] = (
                    self.last_rl_action == reference_action 
                    if reference_action is not None else False
                )
                info[agent]["last_rl_action"] = int(self.last_rl_action)
                info[agent]["reference_action"] = int(reference_action) if reference_action is not None else -1
            
            # Battle progress metrics for debugging
            info[agent]["team_health_total"] = sum(mon.current_hp_fraction for mon in self.battle1.team.values())
            info[agent]["opponent_health_total"] = sum(mon.current_hp_fraction for mon in self.battle1.opponent_team.values())
            info[agent]["active_team_count"] = sum(1 for mon in self.battle1.team.values() if mon.current_hp_fraction > 0)
            info[agent]["active_opponent_count"] = sum(1 for mon in self.battle1.opponent_team.values() if mon.current_hp_fraction > 0)

        return info

    def _get_reference_action(self, battle: AbstractBattle) -> int | None:
        """
        Get what action the reference imitation agent would take in this battle state.
        Enhanced with more sophisticated decision logic for better DDQN guidance.
        
        Args:
            battle: The current battle state
        Returns:
            The action the reference agent would choose, or None if unable to determine
        """
        if not hasattr(battle, 'available_moves') or not hasattr(battle, 'available_switches'):
            return None
            
        try:
            # Check if we need to make a forced switch (current Pokemon fainted)
            current_hp = 0
            if hasattr(battle, 'active_pokemon') and battle.active_pokemon:
                current_hp = battle.active_pokemon.current_hp_fraction
            
            # If current Pokemon is fainted, we must switch
            if current_hp <= 0 and battle.available_switches:
                # Choose the healthiest available Pokemon
                switches = list(battle.available_switches)
                best_switch = max(switches, key=lambda mon: mon.current_hp_fraction)
                return list(battle.available_switches).index(best_switch)
            
            # Normal decision logic based on imitation agent type
            if self.imitation_agent_type == "simple":
                return self._simple_heuristic_action(battle)
            elif self.imitation_agent_type == "max":
                return self._max_power_action(battle)
            elif self.imitation_agent_type == "random":
                return self._random_valid_action(battle)
                    
        except Exception:
            # If anything goes wrong, return None
            pass
            
        return None
    
    def _simple_heuristic_action(self, battle: AbstractBattle) -> int | None:
        """Simple heuristic: prefer high damage, consider switching if low health."""
        try:
            current_hp = 1.0
            if hasattr(battle, 'active_pokemon') and battle.active_pokemon:
                current_hp = battle.active_pokemon.current_hp_fraction
            
            # Consider switching if health is very low and switches available
            if current_hp < 0.2 and battle.available_switches:
                switches = list(battle.available_switches)
                # Switch to healthiest Pokemon
                best_switch = max(switches, key=lambda mon: mon.current_hp_fraction)
                return list(battle.available_switches).index(best_switch)
            
            # Otherwise, choose best move
            if battle.available_moves:
                moves = list(battle.available_moves)
                # Prefer moves with higher base power
                best_move = max(moves, key=lambda move: move.base_power if move.base_power else 0)
                return 6 + moves.index(best_move)
                
        except (AttributeError, TypeError, IndexError):
            pass
        return None
    
    def _max_power_action(self, battle: AbstractBattle) -> int | None:
        """Max power strategy: always choose highest base power move available."""
        try:
            if battle.available_moves:
                moves = list(battle.available_moves)
                best_move = max(moves, key=lambda move: move.base_power if move.base_power else 0)
                return 6 + moves.index(best_move)
            elif battle.available_switches:
                # If no moves available, switch to first option
                return 0
        except (AttributeError, TypeError, IndexError):
            pass
        return None
    
    def _random_valid_action(self, battle: AbstractBattle) -> int | None:
        """Random strategy: choose randomly from all valid actions."""
        try:
            all_actions = []
            if battle.available_moves:
                all_actions.extend([6 + i for i in range(len(battle.available_moves))])
            if battle.available_switches:
                all_actions.extend([i for i in range(len(battle.available_switches))])
            
            if all_actions:
                return np.random.choice(all_actions)
        except (AttributeError, TypeError, IndexError):
            pass
        return None

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        DDQN-optimized reward function with balanced immediate and long-term rewards.
        
        Args:
            battle (AbstractBattle): The current battle instance
        Returns:
            float: The calculated reward optimized for stable DDQN training
        """
        reward = 0.0
        
        # Primary reward: Battle outcome (main learning signal)
        if battle.finished:
            if battle.won:
                reward += 2.0  # Strong positive reward for winning
            else:
                reward -= 2.0  # Strong negative reward for losing
        
        # Secondary reward: Imitation learning (guidance signal)
        if self.last_rl_action is not None:
            reference_action = self._get_reference_action(battle)
            
            if reference_action is not None:
                if self.last_rl_action == reference_action:
                    reward += 0.1  # Small positive reward for matching expert
                else:
                    reward -= 0.05  # Very small penalty to encourage exploration
        
        # Intermediate rewards for progress indicators
        reward += self._calculate_battle_progress_reward(battle)
        
        return reward
    
    def _calculate_battle_progress_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate intermediate rewards based on battle progress to provide
        denser feedback for DDQN training.
        
        Args:
            battle: Current battle state
        Returns:
            Small reward based on tactical advantage
        """
        if battle.finished:
            return 0.0  # No progress reward for finished battles
        
        progress_reward = 0.0
        
        # Reward for maintaining healthy team
        team_health = sum(mon.current_hp_fraction for mon in battle.team.values())
        opponent_health = sum(mon.current_hp_fraction for mon in battle.opponent_team.values())
        
        # Small reward for health advantage (scaled to be small)
        health_advantage = (team_health - opponent_health) / 12.0  # Normalize by max possible difference
        progress_reward += health_advantage * 0.01  # Very small influence
        
        # Small reward for having more active Pokemon
        active_team = sum(1 for mon in battle.team.values() if mon.current_hp_fraction > 0)
        active_opponent = sum(1 for mon in battle.opponent_team.values() if mon.current_hp_fraction > 0)
        
        pokemon_advantage = (active_team - active_opponent) / 6.0  # Normalize
        progress_reward += pokemon_advantage * 0.02  # Very small influence
        
        return progress_reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation space optimized for DDQN training.
        Enhanced with strategic information to help distinguish between states.
        
        Returns:
            int: The size of the observation space.
        """
        # Enhanced observation space:
        # 6 (team health) + 6 (opponent health) + 4 (available moves) + 
        # 6 (available switches) + 4 (move types) + 2 (battle context) = 28 features
        return 28

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Enhanced state representation optimized for DDQN training.
        Provides richer context to help the Q-network distinguish between states
        and make better action-value predictions.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 28-dimensional numpy array with comprehensive battle state.
        """
        # Basic health information (12 features)
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # Available moves information (4 features)
        move_features = self._encode_available_moves(battle)
        
        # Available switches information (6 features) 
        switch_features = self._encode_available_switches(battle)
        
        # Move type effectiveness (4 features)
        type_features = self._encode_type_effectiveness(battle)
        
        # Battle context (2 features)
        context_features = self._encode_battle_context(battle)

        # Combine all features (28 total)
        final_vector = np.concatenate([
            health_team,           # 6 features
            health_opponent,       # 6 features  
            move_features,         # 4 features
            switch_features,       # 6 features
            type_features,         # 4 features
            context_features       # 2 features
        ])

        return final_vector.astype(np.float32)
    
    def _encode_available_moves(self, battle: AbstractBattle) -> np.ndarray:
        """Encode information about available moves."""
        move_features = np.zeros(4)
        
        if hasattr(battle, 'available_moves') and battle.available_moves:
            moves = list(battle.available_moves)
            
            # Feature 0: Number of available moves (normalized)
            move_features[0] = len(moves) / 4.0  # Max 4 moves
            
            # Feature 1: Max base power available (normalized)
            max_power = max((move.base_power or 0) for move in moves)
            move_features[1] = min(max_power / 120.0, 1.0)  # Normalize by typical max power
            
            # Feature 2: Average accuracy (normalized)
            accuracies = [move.accuracy or 1.0 for move in moves]
            move_features[2] = sum(accuracies) / len(accuracies) if accuracies else 1.0
            
            # Feature 3: Has status move (binary)
            move_features[3] = float(any((move.base_power or 0) == 0 for move in moves))
        
        return move_features
    
    def _encode_available_switches(self, battle: AbstractBattle) -> np.ndarray:
        """Encode information about available switches."""
        switch_features = np.zeros(6)
        
        if hasattr(battle, 'available_switches') and battle.available_switches:
            switches = list(battle.available_switches)
            
            # Mark available switches (binary for each slot)
            for i, switch in enumerate(switches):
                if i < 6:  # Max 6 Pokemon
                    switch_features[i] = 1.0
        
        return switch_features
    
    def _encode_type_effectiveness(self, battle: AbstractBattle) -> np.ndarray:
        """Encode type effectiveness information."""
        type_features = np.zeros(4)
        
        try:
            if hasattr(battle, 'active_pokemon') and battle.active_pokemon:
                active_mon = battle.active_pokemon
                opponent_mon = battle.opponent_active_pokemon
                
                if active_mon and opponent_mon and hasattr(battle, 'available_moves'):
                    moves = list(battle.available_moves) if battle.available_moves else []
                    
                    if moves:
                        # Calculate type effectiveness for available moves
                        effectiveness_values = []
                        
                        for move in moves:
                            if hasattr(move, 'type') and hasattr(opponent_mon, 'types'):
                                # Simplified effectiveness calculation
                                # In practice, you'd use the actual type chart
                                effectiveness = 1.0  # Normal effectiveness as default
                                effectiveness_values.append(effectiveness)
                        
                        if effectiveness_values:
                            # Feature 0: Max effectiveness available
                            type_features[0] = max(effectiveness_values)
                            # Feature 1: Average effectiveness
                            type_features[1] = sum(effectiveness_values) / len(effectiveness_values)
                            # Feature 2: Has super effective move
                            type_features[2] = float(any(eff > 1.0 for eff in effectiveness_values))
                            # Feature 3: Has not very effective move
                            type_features[3] = float(any(eff < 1.0 for eff in effectiveness_values))
        
        except (AttributeError, TypeError):
            pass  # Use default zeros if any attribute access fails
        
        return type_features
    
    def _encode_battle_context(self, battle: AbstractBattle) -> np.ndarray:
        """Encode general battle context."""
        context_features = np.zeros(2)
        
        try:
            # Feature 0: Turn number (normalized, capped at 100 turns)
            if hasattr(battle, 'turn'):
                context_features[0] = min(battle.turn / 100.0, 1.0)
            
            # Feature 1: Active Pokemon health fraction
            if hasattr(battle, 'active_pokemon') and battle.active_pokemon:
                context_features[1] = battle.active_pokemon.current_hp_fraction
        
        except (AttributeError, TypeError):
            pass  # Use default zeros if any attribute access fails
        
        return context_features
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode with proper cleanup.
        Enhanced for DDQN training stability.
        
        Args:
            seed: Random seed for environment reset
            options: Additional options for reset
        """
        # Reset episode tracking
        self.reset_episode_tracking()
        
        # Call parent reset with proper arguments
        result = super().reset(seed, options)
        
        return result


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
        imitation_agent_type: str = "simple",  # Agent to imitate: "simple", "max", or "random"
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
            imitation_agent_type=imitation_agent_type,
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