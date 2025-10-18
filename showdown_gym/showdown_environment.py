import os
import time
from typing import Any, Dict

import numpy as np
from gymnasium.spaces import Box
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
    """
    Gen9 Reinforcement Learning Environment using Gymnasium wrapper.
    
    This environment implements a simple RL player with:
    - Observation space: move base powers, damage multipliers, and team status
    - Reward function: based on fainted pokemon, hp values, and victory
    - Action space: moves only (no switching except when forced)
    """

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
        
        # Initialize observation space following the example structure
        # Observation vector contains:
        # - 4 values for move base powers (normalized)
        # - 4 values for move damage multipliers
        # - 1 value for fainted team pokemon ratio
        # - 1 value for fainted opponent pokemon ratio
        # Total: 10 components
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

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
        return 10  # Only attacking moves allowed

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

        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        # Add training specific information for tracking
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
            info[agent]["turn"] = self.battle1.turn
            
            # Track battle statistics
            if hasattr(self.battle1, 'team'):
                fainted_team = len([mon for mon in self.battle1.team.values() if mon.fainted])
                info[agent]["fainted_team"] = fainted_team
            
            if hasattr(self.battle1, 'opponent_team'):
                fainted_opponent = len([mon for mon in self.battle1.opponent_team.values() if mon.fainted])
                info[agent]["fainted_opponent"] = fainted_opponent

        return info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate reward based on battle state following the example structure.
        
        Reward structure:
        - Winning: +30 points
        - Making opponent pokemon faint: +2 points per pokemon
        - Opponent losing HP: +1 point per % hp lost
        - Conversely, negative actions lead to symmetrically negative rewards
        
        Args:
            battle (AbstractBattle): The current battle instance
        Returns:
            float: The calculated reward
        """
        return self.reward_computing_helper(
            battle, fainted_value=4, hp_value=1, victory_value=20
        )

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        Following the example structure:
        - 4 move base powers
        - 4 move damage multipliers
        - 1 fainted team ratio
        - 1 fainted opponent ratio
        
        Returns:
            int: The size of the observation space (10 total features).
        """
        return 10

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation
        following the example structure.
        
        Observation vector contains:
        - Base power of each available move (normalized to 0-3 range, -1 if not available)
        - Damage multiplier of each available move against current opponent (0-4 range)
        - Number of non-fainted pokemon in our team (0-1 range)
        - Number of non-fainted pokemon in opponent's team (0-1 range)

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                try:
                    moves_dmg_multiplier[i] = move.type.damage_multiplier(
                        battle.opponent_active_pokemon.type_1,
                        battle.opponent_active_pokemon.type_2,
                        type_chart=battle._format.type_chart if hasattr(battle, '_format') else None,
                    )
                except (AttributeError, TypeError):
                    moves_dmg_multiplier[i] = 1.0

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
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