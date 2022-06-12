from abc import abstractmethod

from ding.envs import BaseEnv


class BaseGameEnv(BaseEnv):
    """
    Inherit this class for muzero to play
    """

    @abstractmethod
    def current_player(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config.
        """
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
        Returns:
            An array of integers, subset of the action space.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display the game observation.
        """
        pass

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the action to play for the player {self.to_play()}: ")
        while int(choice) not in self.legal_actions():
            choice = input("Ilegal action. Enter another action : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return str(action_number)
