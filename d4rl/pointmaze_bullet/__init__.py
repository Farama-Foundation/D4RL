from ..pointmaze.maze_model import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE, U_MAZE_EVAL, MEDIUM_MAZE_EVAL, LARGE_MAZE_EVAL
from gym.envs.registration import register
from d4rl import infos

register(
    id='bullet-maze2d-open-v0',
    entry_point='d4rl.pointmaze_bullet.bullet_maze:Maze2DBulletEnv',
    max_episode_steps=150,
    kwargs={
        'maze_spec':OPEN,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': infos.REF_MIN_SCORE['bullet-maze2d-open-v0'],
        'ref_max_score': infos.REF_MAX_SCORE['bullet-maze2d-open-v0'],
        'dataset_url':infos.DATASET_URLS['bullet-maze2d-open-v0'],
    }
)

register(
    id='bullet-maze2d-umaze-v0',
    entry_point='d4rl.pointmaze_bullet.bullet_maze:Maze2DBulletEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': infos.REF_MIN_SCORE['bullet-maze2d-umaze-v0'],
        'ref_max_score': infos.REF_MAX_SCORE['bullet-maze2d-umaze-v0'],
        'dataset_url':infos.DATASET_URLS['bullet-maze2d-umaze-v0'],
    }
)

register(
    id='bullet-maze2d-medium-v0',
    entry_point='d4rl.pointmaze_bullet.bullet_maze:Maze2DBulletEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': infos.REF_MIN_SCORE['bullet-maze2d-medium-v0'],
        'ref_max_score': infos.REF_MAX_SCORE['bullet-maze2d-medium-v0'],
        'dataset_url':infos.DATASET_URLS['bullet-maze2d-medium-v0'],
    }
)

register(
    id='bullet-maze2d-large-v0',
    entry_point='d4rl.pointmaze_bullet.bullet_maze:Maze2DBulletEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': infos.REF_MIN_SCORE['bullet-maze2d-large-v0'],
        'ref_max_score': infos.REF_MAX_SCORE['bullet-maze2d-large-v0'],
        'dataset_url':infos.DATASET_URLS['bullet-maze2d-large-v0'],
    }
)
