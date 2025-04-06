from .sokoban import SokobanEnv,SokobanConfig
from .frozenlake import FrozenLakeEnv,FrozenLakeConfig
from .navigation import NavigationEnv, NavigationConfig
REGISTERED_ENV = {
    "sokoban": {
        "env": SokobanEnv,
        "config": SokobanConfig,
    },
    "frozenlake": {
        "env": FrozenLakeEnv,
        "config": FrozenLakeConfig,
    },
    "navigation": {
        "env": NavigationEnv,
        "config": NavigationConfig,
    }
}