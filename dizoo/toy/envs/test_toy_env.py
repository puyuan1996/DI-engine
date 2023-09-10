import numpy as np
import pytest
from toy_env import ToyEnv  # replace 'your_module' with the actual module name


def test_toy_env():
    # Initialize environment
    env = ToyEnv({})

    # Test reset
    state = env.reset()
    assert np.sum(state) == 1
    assert np.where(state == 1)[0][0] in [0, 1, 2, 3]

    # Test step
    for _ in range(100):
        action = env.action_space.sample()  # random action
        timestep = env.step(action)
        assert np.sum(timestep.obs) == 1
        assert np.where(timestep.obs == 1)[0][0] in [0, 1, 2, 3]
        assert timestep.reward in [0, 1]
        assert isinstance(timestep.done, bool)
        assert timestep.done is True or timestep.done is False


    # Test transitions, rewards and done
    env.reset(np.array([0]))
    timestep = env.step(np.array([0.75, 0.90]))  # 0.9375 is in the threshold of the action space
    assert (timestep.obs == np.array([1, 0, 0, 0])).all
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([0]))
    timestep = env.step(np.array([0.75, 0.94]))  # 0.9375 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 1, 0, 0])).all
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([1]))
    timestep = env.step(np.array([0.75, 0.90]))  # 0.9375 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 1, 0, 0])).all
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([1]))
    timestep = env.step(np.array([0.75, 0.94]))  # 0.9375 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 1, 0])).all
    assert timestep.reward == 0
    assert timestep.done is False

    env.reset(np.array([2]))
    timestep = env.step(np.array([2.875, 3.1]))  # 3.0625 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 0, 1])).all

    assert timestep.reward == 1
    assert timestep.done is True

    env.reset(np.array([2]))
    timestep = env.step(np.array([2.875, 3.0]))  # 3.0625 is in the threshold of the action space
    assert (timestep.obs == np.array([0, 0, 1, 0])).all
    assert timestep.reward == 0
    assert timestep.done is False

    # Test close
    env.close()  # no assert statement required as close doesn't return anything

# if the script is run directly, run the tests
if __name__ == "__main__":
    pytest.main([__file__])