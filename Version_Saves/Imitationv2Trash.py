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

    def _get_action_size(self) -> int | None:
        """
        Returns the size of the action space for the agent.

        Modified action space with only attacking moves:
        - 4 regular moves (0-3)
        - 4 mega evolve moves (4-7) 
        - 4 z-moves (8-11)
        - 4 dynamax moves (12-15)
        - 4 terastallize moves (16-19)
        
        Total: 20 actions (reduced from original 26 by removing 6 switch actions)
        """
        return 20  # Only attacking moves allowed

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        Modified action mapping - only allows attacking moves:
        action = -2: default (forced switch when Pokemon faints)
        action = -1: forfeit
        0 <= action <= 3: move (attacking moves only)
        4 <= action <= 7: move and mega evolve (attacking moves only)
        8 <= action <= 11: move and z-move (attacking moves only)
        12 <= action <= 15: move and dynamax (attacking moves only)
        16 <= action <= 19: move and terastallize (attacking moves only)

        Switching (actions 0-5 from original) is only allowed when Pokemon faints.

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        # Store the action for imitation learning reward calculation
        self.last_rl_action = action
        
        # Check if current Pokemon is fainted and we need to make a forced switch
        if hasattr(self, 'battle1') and self.battle1 is not None:
            battle = self.battle1
            
            # If Pokemon is fainted, force a switch (return -2 for default switch)
            if (battle.active_pokemon is None or 
                (hasattr(battle.active_pokemon, 'fainted') and battle.active_pokemon.fainted) or
                (hasattr(battle.active_pokemon, 'current_hp') and battle.active_pokemon.current_hp == 0)):
                return -2  # Default action for forced switch
            
            # Otherwise, only allow attacking moves
            # Map actions to move indices (6-9 in original mapping become 0-3 here)
            if 0 <= action <= 3:  # Regular moves
                return action + 6  # Convert to original move indices (6-9)
            elif 4 <= action <= 7:  # Mega evolve moves
                return action + 6  # Convert to original indices (10-13)
            elif 8 <= action <= 11:  # Z-moves
                return action + 6  # Convert to original indices (14-17)
            elif 12 <= action <= 15:  # Dynamax moves
                return action + 6  # Convert to original indices (18-21)
            elif 16 <= action <= 19:  # Terastallize moves
                return action + 6  # Convert to original indices (22-25)
            else:
                # Default to first available move if action is out of range
                return 6
        
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add imitation learning specific information for tracking
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["imitation_agent_type"] = self.imitation_agent_type
            
            # Track if the last action matched the reference agent
            if hasattr(self, 'last_rl_action') and self.last_rl_action is not None:
                reference_action = self._get_reference_action(self.battle1)
                info[agent]["action_match"] = (
                    self.last_rl_action == reference_action 
                    if reference_action is not None else False
                )
                info[agent]["last_rl_action"] = int(self.last_rl_action)
                info[agent]["reference_action"] = int(reference_action) if reference_action is not None else -1

        return info

    def _get_reference_action(self, battle: AbstractBattle) -> int | None:
        """
        Get what action the reference imitation agent would take in this battle state.
        Modified to only use attacking moves, following the same constraints as the RL agent.
        
        Args:
            battle: The current battle state
        Returns:
            The action the reference agent would choose, or None if unable to determine
        """
        if not hasattr(battle, 'available_moves'):
            return None
            
        try:
            # Only consider attacking moves - no switching allowed except when forced
            if battle.available_moves:
                if self.imitation_agent_type == "simple" or self.imitation_agent_type == "max":
                    # Both SimpleHeuristicsPlayer and MaxBasePowerPlayer prefer highest base power
                    best_move = max(battle.available_moves, 
                                   key=lambda move: move.base_power if move.base_power else 0)
                    # Convert to our new action mapping (0-3 for regular moves)
                    move_index = list(battle.available_moves).index(best_move)
                    return min(move_index, 3)  # Limit to first 4 moves (0-3)
                    
                elif self.imitation_agent_type == "random":
                    # For random, choose from available attacking moves only
                    num_moves = min(len(battle.available_moves), 4)  # Max 4 moves
                    return np.random.randint(0, num_moves) if num_moves > 0 else 0
            else:
                # No moves available, return default action
                return 0
                    
        except Exception:
            # If anything goes wrong, return default move
            return 0
            
        return 0

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Imitation Learning Reward: Agent gets positive reward for choosing the same
        action as the reference agent, negative reward for different actions.

        Args:
            battle (AbstractBattle): The current battle instance
        Returns:
            float: The calculated reward based on action similarity to reference agent
        """
        reward = 0.0
        
        # Only calculate imitation reward if we have stored the agent's last action
        if self.last_rl_action is not None:
            reference_action = self._get_reference_action(battle)
            
            if reference_action is not None:
                # Give positive reward for matching the reference agent's choice
                if self.last_rl_action == reference_action:
                    reward += 0.5  # Match reward
                else:
                    reward -= 0.1  # Mismatch penalty (smaller to encourage exploration)`
                    
        # Add battle outcome rewards normalized by battle length (turn count)
        if battle.finished:
            # Get current turn count, ensure it's at least 1 to avoid division by zero
            turn_count = max(battle.turn, 1)
            
            if battle.won:
                reward += 10 / turn_count  # Win bonus normalized by turns
            else:
                reward -= 0.5 * turn_count  # Loss penalty normalized by turns
        
        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        Enhanced observation space includes:
        - Team health (6) + Opponent health (6) = 12
        - Active Pokemon type effectiveness (18 types) = 18  
        - Turn count (1) = 1
        
        Returns:
            int: The size of the observation space (31 total features).
        """
        # Total: 12 + 18 + 1 = 31 features
        return 31

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        
        Enhanced state representation includes health, type effectiveness, turn count,
        field conditions, and Pokemon stats to provide richer information for learning.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        # 1. Health information (12 features)
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # 2. Type effectiveness for active Pokemon (18 features - one for each type)
        type_effectiveness = self._get_type_effectiveness(battle)
        
        # 3. Turn count (1 feature) - normalized by dividing by 100 to keep values reasonable
        turn_count = [min(battle.turn / 100.0, 1.0)]

        # Combine all features into final observation vector (31 total)
        final_vector = np.concatenate([
            health_team,           # 6 features
            health_opponent,       # 6 features  
            type_effectiveness,    # 18 features
            turn_count            # 1 feature
        ])

        return final_vector.astype(np.float32)
    
    def _get_type_effectiveness(self, battle: AbstractBattle) -> np.ndarray:
        """Get type effectiveness multipliers for the active Pokemon against opponent types."""
        # Pokemon types in standard order
        types = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy']
        
        effectiveness = np.ones(18)  # Default 1.0 (neutral effectiveness)
        
        if battle.active_pokemon and battle.opponent_active_pokemon:
            active_mon = battle.active_pokemon
            opponent_mon = battle.opponent_active_pokemon
            
            # Get effectiveness of our active Pokemon's moves against opponent
            if hasattr(active_mon, 'moves') and opponent_mon.types:
                for i, type_name in enumerate(types):
                    # Calculate average effectiveness against opponent's types
                    total_effectiveness = 0.0
                    move_count = 0
                    
                    for move in active_mon.moves:
                        if move and hasattr(move, 'type'):
                            for opp_type in opponent_mon.types:
                                # This is a simplified effectiveness calculation
                                # In reality, you'd need the full type chart
                                if move.type.name.lower() == type_name:
                                    if opp_type.name.lower() in ['fire', 'water', 'grass']:  # Example
                                        total_effectiveness += 2.0 if type_name == 'water' and opp_type.name.lower() == 'fire' else 1.0
                                    else:
                                        total_effectiveness += 1.0
                                    move_count += 1
                    
                    if move_count > 0:
                        effectiveness[i] = total_effectiveness / move_count
        
        return effectiveness


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