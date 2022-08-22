import os
import json

from gym.envs.registration import register, registry


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


for scale in [100, 200]:
    # filter urls with tsp{scales}
    prefix = f"TSP{scale}"

    # load urls
    with open(os.path.join(CUR_DIR, f"tsp{scale}-expert-info.json"), 'r') as f:
        data = json.load(f)
        DATASET_URLS = data["DATASET_URLS"]
        REF_MIN_SCORES = data["REF_MIN_SCORES"]
        REF_MAX_SCORES = data["REF_MAX_SCORES"]

    with open(os.path.join(CUR_DIR, f"tsp{scale}-action-dim.json"), 'r') as f:
        action_dims = json.load(f)

    file_uris = [uri for key, uri in DATASET_URLS.items()]
    seeds = [int(key.split("/")[-1]) for key in DATASET_URLS.keys()]

    for seed, file_uri in zip(seeds, file_uris):
        env_id = f'tsp{scale}-{seed}-expert-v1'
        if env_id in registry.env_specs:
            continue
        register(
            id=env_id,
            entry_point='d4rl.tsp.env:OfflineTSPEnv',
            max_episode_steps=scale,
            kwargs={
                'scale': scale,
                'env_dataset_path': f"/nfs/dgx5/raid/gato_dataset/tsp{scale}/tsp{scale}_train_seed{seed}.pkl",
                'env_node_embedding_path': f"/nfs/dgx5/raid/gato_dataset/tsp{scale}/tsp{scale}_{seed}_node_embeddings.npy",
                'action_dim': int(action_dims[str(seed)]),
                'dataset_url': file_uri,
                'ref_min_score': REF_MIN_SCORES[f"{prefix}/{seed}"],
                'ref_max_score': REF_MAX_SCORES[f"{prefix}/{seed}"],
                'legal_keys': [
                    'observations', 'actions', 'rewards', 'terminals', 'ava_actions'
                ]
            }
        )


if __name__ == "__main__":
    import gym
    import time
    import numpy as np

    scale = 100
    seed = 66
    env = gym.make(f"tsp{scale}-{seed}-expert-v1")
    
    dataset = env.get_dataset()
    n_step = 0
    n_episode = 10
    start_time = time.time()

    for episode_th in range(n_episode):
        obs = env.reset()
        done = False
        n_step = 0
        total_reward = 0

        while not done:
            ava_idxes = [idx for idx, v in enumerate(env.get_cur_action_mask()) if v == 1]
            node_idx = np.random.choice(ava_idxes)
            obs, reward, done, info = env.step(node_idx)
            total_reward += reward
            n_step += 1
            # print(f"\t* step: {n_step}, eposide: {episode_th}, action: {node_idx}, ava_actions: {len(ava_idxes)} rew: {reward}, info: {info}, obs_shape: {obs.shape}")

            # if n_step % 10 == 0:
            #     fsp = n_step / (time.time() - start_time)
            #     print(f"episode_th: {episode_th}, n_step: {n_step}, FPS: {fsp}")
        print(f"episode_th: {episode_th}, total_reward: {total_reward}")
