from pathlib import Path
from gym.envs.registration import register

# CARLA_GYM_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CARLA_GYM_ROOT_DIR = Path(__file__).resolve().parent

_Available_Envs = {
    'LeaderBoard-v0': {
        'entry_point': 'carla_gym.leaderboard_env:LeaderboardEnv',
        'description': 'leaderboard route with no-that-dense backtround traffic',
        'kwargs': {}
    }
}

for env_id, val in _Available_Envs.items():
    register(id=env_id, entry_point=val.get('entry_point'), kwargs=val.get('kwargs'))


if __name__ == '__main__':
    print(CARLA_GYM_ROOT_DIR)
    # /oldhome/yihang/MECarla/carla_gym
