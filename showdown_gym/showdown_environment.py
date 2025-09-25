# Provides access to operating system functionality
import os
# For type hinting: Any type and dictionary
from typing import Any, Dict

# For numerical operations (not used directly in this snippet)
import numpy as np
# Import different player types from poke_env for battle strategies
from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
# AbstractBattle class for type hinting and battle logic
from poke_env.battle import AbstractBattle
# Wrapper for single-agent environments
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
# Base player class from poke_env
from poke_env.player.player import Player

# Import the base environment for Showdown from local module
from showdown_gym.base_environment import BaseShowdownEnv

# Main environment class for Pokémon Showdown RL tasks
class ShowdownEnvironment(BaseShowdownEnv):

    def __init__(
        self,
        battle_format: str = "gen9randombattle",  # Format of the battle (e.g., Gen 9 random battle)
        account_name_one: str = "train_one",      # Name for the first account (agent)
        account_name_two: str = "train_two",      # Name for the second account (opponent)
        team: str | None = None,                   # Optional team string (if not random)
    ):
        # Initialize the base environment with the provided settings
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        # Get the base info dictionary from the parent class
        info = super().get_additional_info()

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        # If a battle has occurred, record whether the agent won
        if self.battle1 is not None:
            agent = self.possible_agents[0]  # Get the agent's name
            info[agent]["win"] = self.battle1.won  # Store win status (True/False)

            # --- Additional Pokémon Showdown info for reward engineering ---
            battle = self.battle1

            # Damage dealt to opponent (sum of max_hp - current_hp for all opponent Pokémon)
            info[agent]["damage_dealt"] = float(sum([mon.max_hp - mon.current_hp for mon in battle.opponent_team.values() if mon.max_hp is not None and mon.current_hp is not None]))

            # Damage taken by agent (sum of max_hp - current_hp for all agent Pokémon)
            info[agent]["damage_taken"] = float(sum([mon.max_hp - mon.current_hp for mon in battle.team.values() if mon.max_hp is not None and mon.current_hp is not None]))

            # Turns for the battle
            info[agent]["turns"] = battle.turn

            # Number of Pokémon left over (not fainted) for agent and opponent
            info[agent]["pokemon_left"] = sum([not mon.fainted for mon in battle.team.values()])
            info[agent]["opponent_pokemon_left"] = sum([not mon.fainted for mon in battle.opponent_team.values()])

        return info  # Return the info dictionary with any additional info

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculates the reward based on the changes in state of the battle.

        You need to implement this method to define how the reward is calculated

        Args:
            battle (AbstractBattle): The current battle instance containing information
                about the player's team and the opponent's team from the player's perspective.
            prior_battle (AbstractBattle): The prior battle instance to compare against.
        Returns:
            float: The calculated reward based on the change in state of the battle.
        """

        prior_battle = self._get_prior_battle(battle)

        reward = 0.0

        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [
            mon.current_hp_fraction for mon in battle.opponent_team.values()
        ]

        # If the opponent has less than 6 Pokémon, fill the missing values with 1.0 (fraction of health)
        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        prior_health_opponent = []
        if prior_battle is not None:
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]

        # Ensure health_opponent has 6 components, filling missing values with 1.0 (fraction of health)
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend(
                [1.0] * (len(health_team) - len(prior_health_opponent))
            )

        diff_health_opponent = np.array(prior_health_opponent) - np.array(
            health_opponent
        )

        # Reward for reducing the opponent's health
        reward += np.sum(diff_health_opponent)

        # Fainting bonus: +2 for each opponent Pokémon fainted since last step
        prior_fainted_opponent = 0
        if prior_battle is not None:
            prior_fainted_opponent = sum([mon.fainted for mon in prior_battle.opponent_team.values()])
        current_fainted_opponent = sum([mon.fainted for mon in battle.opponent_team.values()])
        fainted_diff = current_fainted_opponent - prior_fainted_opponent
        reward += 2.0 * fainted_diff

        # Survival bonus: +1 for each of agent's Pokémon still alive at the end of the battle
        if battle.finished:
            agent_alive = sum([not mon.fainted for mon in battle.team.values()])
            reward += 1.0 * agent_alive

        # Win/loss bonus: +10 for win, -10 for loss (only at end of battle)
        if battle.finished:
            if battle.won:
                reward += 10.0
            else:
                reward -= 10.0

        return reward

    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 12

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Embeds the current state of a Pokémon battle into a numerical vector representation.
        This method generates a feature vector that represents the current state of the battle,
        this is used by the agent to make decisions.

        You need to implement this method to define how the battle state is represented.

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

        #########################################################################################################
        # Caluclate the length of the final_vector and make sure to update the value in _observation_size above #
        #########################################################################################################

        # Final vector - single array with health of both teams
        final_vector = np.concatenate(
            [
                health_team,  # N components for the health of each pokemon
                health_opponent,  # N components for the health of opponent pokemon
            ]
        )

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
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer()
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer()
        elif opponent_type == "random":
            opponent = RandomPlayer()
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "train_one" if not evaluation else "eval_one"
        account_name_two: str = "train_two" if not evaluation else "eval_two"

        account_name_one = f"{account_name_one}_{opponent_type}"
        account_name_two = f"{account_name_two}_{opponent_type}"

        team = self._load_team(team_type)

        battle_fomat = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_fomat,
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
