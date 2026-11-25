try:
    import gymnasium as gym
except ImportError:
    gym = None


def create_environment():
    if gym is None:
        raise RuntimeError("Gymnasium is not installed")
    raise NotImplementedError("Add the RTL optimization Gymnasium environment")
