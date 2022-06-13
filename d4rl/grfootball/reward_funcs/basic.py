import numpy as np


class Rewarder:
    def __init__(self) -> None:
        self.player_last_hold_ball = -1

    def calc_reward(self, rew, prev_obs, obs):
        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]

        reward = (
            5.0 * win_reward(obs)
            + 5.0 * preprocess_score(obs, rew, self.player_last_hold_ball)
            + 0.003 * ball_position_reward(obs, self.player_last_hold_ball)
            + yellow_reward(prev_obs, obs)
            - 0.003 * min_dist_reward(obs)
            # + lost_ball_reward(prev_obs, obs, self.player_last_hold_ball)
        )

        return reward


def preprocess_score(obs, rew_signal, player_last_hold_ball):
    if rew_signal > 0:
        factor = 1.0  # if obs["active"] == player_last_hold_ball else 0.3
    else:
        return rew_signal
    return rew_signal * factor


def lost_ball_reward(prev_obs, obs, player_last_hold_ball):
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
        if obs["active"] == player_last_hold_ball:
            return -0.5
    return -0.1


def win_reward(obs):
    win_reward = 0.0
    # print(f"steps left: {obs['steps_left']}")
    if obs["steps_left"] == 0:
        # print("STEPS LEFT == 0!")
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = my_score - opponent_score
    return win_reward


def min_dist_reward(obs):
    if obs["ball_owned_team"] != 0:
        ball_position = np.array(obs["ball"][:2])
        left_team_position = obs["left_team"][1:]
        left_team_dist2ball = np.linalg.norm(left_team_position - ball_position, axis=1)
        min_dist2ball = np.min(left_team_dist2ball)
    else:
        min_dist2ball = 0.0
    return min_dist2ball


def yellow_reward(prev_obs, obs):
    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow
    return yellow_r


def ball_position_reward(obs, player_last_hold_ball):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -2.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -1.0
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 2.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 1.0
    else:
        ball_position_r = 0.0

    # if obs["ball_owned_team"] == 0:
    #     if not obs["active"] == player_last_hold_ball:
    #         ball_position_r *= 0.5

    return ball_position_r