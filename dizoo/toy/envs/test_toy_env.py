import numpy as np
import pytest
from easydict import EasyDict
from toy_env import ToyEnv  # replace 'your_module' with the actual module name


def test_toy_env():
    # Initialize environment
    env = ToyEnv( EasyDict(dict(act_transform=False, max_episode_length=10)) )

    # Test reset
    state = env.reset()
    assert np.sum(state) == 1
    assert np.where(state == 1)[0][0] in [0, 1, 2, 3]

    test_actions = [[0.75, 0.94], [0.75, 0.94], [0.8929, 3.1194]]
    for i in range(3):

        timestep = env.step(test_actions[i])
        assert np.sum(timestep.obs) == 1
        assert np.where(timestep.obs == 1)[0][0] in [0, 1, 2, 3]
        assert timestep.reward in [0, 1]
        assert isinstance(timestep.done, bool)
        if i == 2:
            assert timestep.done is True
            assert timestep.reward == 1
        else:
            assert timestep.done is False
            assert timestep.reward == 0



    # Test transitions, rewards and done
    env.reset(np.array([0]))
    timestep = env.step(np.array([0.75, 0.90]))  # 0.9375 is the threshold of the action space
    assert (timestep.obs == np.array([1, 0, 0, 0])).all()
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([0]))
    timestep = env.step(np.array([0.75, 0.94]))  # 0.9375 is  the threshold of the action space
    assert (timestep.obs == np.array([0, 1, 0, 0])).all()
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([1]))
    timestep = env.step(np.array([0.75, 0.90]))  # 0.9375 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 1, 0, 0])).all()
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([1]))
    timestep = env.step(np.array([0.75, 0.94]))  # 0.9375 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 1, 0])).all()
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([2]))
    timestep = env.step(np.array([2.875, 3.1]))  # 3.0625 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 0, 1])).all()

    assert timestep.reward == 1
    assert timestep.done is True

    env.reset(np.array([2]))
    timestep = env.step(np.array([2.875, 3.0]))  # 3.0625 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 1, 0])).all()
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([2]))
    timestep = env.step(np.array([0.8929, 3.1194]))  # 3.0625 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 0, 1])).all()
    assert timestep.reward == 1
    assert timestep.done is True

    env.reset(np.array([2]))
    timestep = env.step([8928997, 3.1194117])  # 3.0625 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 0, 1])).all()
    assert timestep.reward == 1
    assert timestep.done is True
    # Test close
    env.close()  # no assert statement required as close doesn't return anything

# if the script is run directly, run the tests
if __name__ == "__main__":
    pytest.main([__file__])