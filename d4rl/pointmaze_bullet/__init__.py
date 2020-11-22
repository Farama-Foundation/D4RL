from ..pointmaze.maze_model import OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE, U_MAZE_EVAL, MEDIUM_MAZE_EVAL, LARGE_MAZE_EVAL
from gym.envs.registration import register

register(
    id='bullet-maze2d-open-v0',
    entry_point='d4rl.pointmaze_bullet.bullet_maze:Maze2DBulletEnv',
    max_episode_steps=150,
    kwargs={
        'maze_spec':OPEN,
        'reward_type':'sparse',
        'reset_target': False,
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
    }
)
