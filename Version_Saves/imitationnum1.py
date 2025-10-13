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

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        # Store the action for imitation learning reward calculation
        self.last_rl_action = action
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
        
        Args:
            battle: The current battle state
        Returns:
            The action the reference agent would choose, or None if unable to determine
        """
        if not hasattr(battle, 'available_moves') or not hasattr(battle, 'available_switches'):
            return None
            
        try:
            # Create a temporary reference player to get its action choice
            if self.imitation_agent_type == "simple":
                # SimpleHeuristicsPlayer: prefers high damage moves
                if battle.available_moves:
                    # Choose the move with highest base power
                    best_move = max(battle.available_moves, 
                                   key=lambda move: move.base_power if move.base_power else 0)
                    # Convert to action index (moves start at index 6)
                    return 6 + list(battle.available_moves).index(best_move)
                elif battle.available_switches:
                    # Switch to first available Pokemon (switches are indices 0-5)
                    return list(battle.available_switches).index(battle.available_switches[0])
                    
            elif self.imitation_agent_type == "max":
                # MaxBasePowerPlayer: always chooses highest base power move
                if battle.available_moves:
                    best_move = max(battle.available_moves, 
                                   key=lambda move: move.base_power if move.base_power else 0)
                    return 6 + list(battle.available_moves).index(best_move)
                elif battle.available_switches:
                    return list(battle.available_switches).index(battle.available_switches[0])
                    
            elif self.imitation_agent_type == "random":
                # For random, we'll just return a random valid action
                all_actions = []
                if battle.available_moves:
                    all_actions.extend([6 + i for i in range(len(battle.available_moves))])
                if battle.available_switches:
                    all_actions.extend([i for i in range(len(battle.available_switches))])
                if all_actions:
                    return np.random.choice(all_actions)
                    
        except Exception:
            # If anything goes wrong, return None
            pass
            
        return None

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
                    reward += 1.0  # Match reward
                else:
                    reward -= 0.1  # Mismatch penalty (smaller to encourage exploration)
        
        # Optional: Add small battle outcome rewards to provide additional guidance
        if battle.finished:
            if battle.won:
                reward += 4  # Small win bonus
            else:
                reward -= 4  # Small loss penalty
        
        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        For imitation learning, we keep it simple with just health information.
        
        Returns:
            int: The size of the observation space.
        """
        # 6 for team health + 6 for opponent health = 12 total features
        return 12

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        
        For imitation learning, we keep the state representation simple to speed up training.
        Only health information is used, which is sufficient for the agent to learn basic
        battle patterns from the reference agent.

        Args:
            battle (AbstractBattle): The current battle instance containing information about
                the player's team and the opponent's team.
        Returns:
            np.float32: A 1D numpy array containing the state you want the agent to observe.
        """

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # Simple 12-dimensional observation: 6 team health + 6 opponent health
        # This minimal state representation enables faster training while still
        # providing essential information for the imitation learning agent
        final_vector = np.concatenate([health_team, health_opponent])

        return final_vector


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