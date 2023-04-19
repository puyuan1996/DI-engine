import numpy as np
from zoo.game_2048.envs.game_2048_env import Game2048Env, IllegalMove
import pytest
from easydict import EasyDict

config = EasyDict(dict(
    env_name="game_2048_env_2048",
    save_replay_gif=False,
    replay_path_gif=None,
    replay_path=None,
    act_scale=True,
    channel_last=False,
    obs_type='raw_observation',  # options=['raw_observation', 'dict_observation']
    reward_normalize=True,
    max_tile=2048,
    delay_reward_step=0,
    prob_random_agent=0.,
    max_episode_steps=int(1e4),
))


class Test2048Logic():
    def test_combine(self):
        game_2048_env = Game2048Env(config)
        game_2048_env.reset()
        print('init board state: ')
        # game_2048_env.render()
        # Test not combining
        assert game_2048_env.combine([0, 0, 0, 0]) == ([0, 0, 0, 0], 0)
        assert game_2048_env.combine([2, 4, 8, 16]) == ([2, 4, 8, 16], 0)

        # Test combining
        # Left same same
        assert game_2048_env.combine([2, 2, 8, 0]) == ([4, 8, 0, 0], 4)
        # Middle the same
        assert game_2048_env.combine([4, 2, 2, 4]) == ([4, 4, 4, 0], 4)
        # Left and middle the same
        assert game_2048_env.combine([2, 2, 2, 8]) == ([4, 2, 8, 0], 4)
        # Right the same
        assert game_2048_env.combine([2, 8, 4, 4]) == ([2, 8, 8, 0], 8)
        # Left and right the same
        assert game_2048_env.combine([2, 2, 4, 4]) == ([4, 8, 0, 0], 12)
        # Right and middle the same
        assert game_2048_env.combine([2, 4, 4, 4]) == ([2, 8, 4, 0], 8)
        # All the same
        assert game_2048_env.combine([4, 4, 4, 4]) == ([8, 8, 0, 0], 16)

        # Test short input
        assert game_2048_env.combine([]) == ([0, 0, 0, 0], 0)
        assert game_2048_env.combine([0]) == ([0, 0, 0, 0], 0)
        assert game_2048_env.combine([2]) == ([2, 0, 0, 0], 0)
        assert game_2048_env.combine([2, 4]) == ([2, 4, 0, 0], 0)
        assert game_2048_env.combine([2, 2, 8]) == ([4, 8, 0, 0], 4)

    def test_shift(self):
        game_2048_env = Game2048Env(config)
        game_2048_env.reset()
        # print('init board state: ')
        # game_2048_env.render()
        # Shift left without combining
        assert game_2048_env.shift([0, 0, 0, 0], 0) == ([0, 0, 0, 0], 0)
        assert game_2048_env.shift([0, 2, 0, 0], 0) == ([2, 0, 0, 0], 0)
        assert game_2048_env.shift([0, 2, 0, 4], 0) == ([2, 4, 0, 0], 0)
        assert game_2048_env.shift([2, 4, 8, 16], 0) == ([2, 4, 8, 16], 0)

        # Shift left and combine
        assert game_2048_env.shift([0, 2, 2, 8], 0) == ([4, 8, 0, 0], 4)
        assert game_2048_env.shift([2, 2, 2, 8], 0) == ([4, 2, 8, 0], 4)
        assert game_2048_env.shift([2, 2, 4, 4], 0) == ([4, 8, 0, 0], 12)

        # Shift right without combining
        assert game_2048_env.shift([0, 0, 0, 0], 1) == ([0, 0, 0, 0], 0)
        assert game_2048_env.shift([0, 2, 0, 0], 1) == ([0, 0, 0, 2], 0)
        assert game_2048_env.shift([0, 2, 0, 4], 1) == ([0, 0, 2, 4], 0)
        assert game_2048_env.shift([2, 4, 8, 16], 1) == ([2, 4, 8, 16], 0)

        # Shift right and combine
        assert game_2048_env.shift([2, 2, 8, 0], 1) == ([0, 0, 4, 8], 4)
        assert game_2048_env.shift([2, 2, 2, 8], 1) == ([0, 2, 4, 8], 4)
        assert game_2048_env.shift([2, 2, 4, 4], 1) == ([0, 0, 4, 8], 12)

    def test_move(self):
        # Test a bunch of lines all moving at once.
        game_2048_env = Game2048Env(config)
        game_2048_env.reset()
        print('init board state: ')
        game_2048_env.render()
        
        # Test shift up
        game_2048_env.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game_2048_env.move(0) == 12
        assert np.array_equal(game_2048_env.get_board(), np.array([
            [4, 4, 8, 4],
            [2, 4, 2, 8],
            [0, 0, 4, 4],
            [0, 0, 0, 0]]))

        # Test shift right
        game_2048_env.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game_2048_env.move(1) == 20
        assert np.array_equal(game_2048_env.get_board(), np.array([
            [0, 0, 2, 4],
            [0, 0, 4, 8],
            [0, 2, 4, 8],
            [0, 0, 4, 8]]))

        # Test shift down
        game_2048_env.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game_2048_env.move(2) == 12
        assert np.array_equal(game_2048_env.get_board(), np.array([
            [0, 0, 0, 0],
            [0, 0, 8, 4],
            [2, 4, 2, 8],
            [4, 4, 4, 4]]))

        # Test shift left
        game_2048_env.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert game_2048_env.move(3) == 20
        assert np.array_equal(game_2048_env.get_board(), np.array([
            [2, 4, 0, 0],
            [4, 8, 0, 0],
            [4, 2, 8, 0],
            [4, 8, 0, 0]]))

        # Test that doing the same move again (without anything added) is illegal
        with pytest.raises(IllegalMove):
            game_2048_env.move(3)

        # Test a follow-on move from the first one
        assert game_2048_env.move(2) == 8  # shift down
        assert np.array_equal(game_2048_env.get_board(), np.array([
            [0, 4, 0, 0],
            [2, 8, 0, 0],
            [4, 2, 0, 0],
            [8, 8, 8, 0]]))

    def test_highest(self):
        game_2048_env = Game2048Env(config)
        game_2048_env.reset()
        # print('init board state: ')
        # game_2048_env.render()
        game_2048_env.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2048, 8],
            [2, 2, 4, 4]]))
        assert game_2048_env.highest() == 2048

    def test_is_end(self):
        game_2048_env = Game2048Env(config)
        game_2048_env.reset()
        # print('init board state: ')
        # game_2048_env.render()
        game_2048_env.set_board(np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]]))
        assert game_2048_env.is_end() is False
        
        game_2048_env.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 8]]))
        assert game_2048_env.is_end() is True


if __name__ == "__main__":
    game_2048_env = Game2048Env(config)
    game_2048_env.reset()
    print('init board state: ')
    game_2048_env.render()
    step = 0
    while True:
        # action = env.human_to_action()
        action = game_2048_env.random_action()
        obs, reward, done, info = game_2048_env.step(action)
        game_2048_env.render()
        step += 1
        print(f"step: {step}, action: {action}, reward: {reward}, raw_reward: {info['raw_reward']}")
        if done:
            print('total_step_number: {}'.format(step))
            break
