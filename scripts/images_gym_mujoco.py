"""
Script for generating image observations for GymMuJoCo and Adroit domains.

To use adroit, set the --adroit flag. Adroit has a different state
loading mechanism from Gym-MuJoCo.

"""
import d4rl
import h5py
import gym
import argparse
import tqdm
import os
import numpy as np
import io
from PIL import Image

from d4rl.utils.h5util import get_keys

def to_bytes(img, quality=95):
    img = Image.fromarray(np.uint8(img))
    bytesio = io.BytesIO()
    img.save(bytesio, format='jpeg', quality=quality)
    img_bytes = bytesio.getvalue()
    #img_bytes = np.frombuffer(img_bytes, dtype=np.int8)
    return img_bytes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--adroit', action='store_true')
    parser.add_argument('--jpg', action='store_true')
    parser.add_argument('--jpg_quality', type=int, default=95)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    env = gym.make(args.env)
    if args.adroit:
        env.offscreen_viewer_setup()
    env.reset()

    dset = h5py.File(args.dataset, 'r')
    N = dset['observations'].shape[0]
    print(dset.keys())
    print(dset['infos'])
    info_keys = list(dset['infos'].keys())
    info_keys = [k for k in info_keys if k != 'action_log_probs']
    print('Info keys:', info_keys)
    state_datasets = {k: dset['infos/'+k][:] for k in info_keys}

    # Copy over other datasets
    path = os.path.split(args.dataset)
    #out_file = os.path.join(path[:-1] + ['images-'+path[-1]])
    out_file = 'images-'+path[-1]
    out_dataset = h5py.File(out_file, 'w')
    for k in get_keys(dset):
        if 'observations' not in k:
            data = dset[k]
            if len(data.shape) > 0:
                data = data[:]
                out_dataset.create_dataset(k, data=data, compression='gzip')
            else:
                data = data[()]
                out_dataset[k] = data


    #total_shape = (dset['observations'].shape[0], )
    #layout = h5py.VirtualLayout(shape=total_shape, dtype=np.string_)
    layout = None

    def create_virtual(index, data, offset, layout):
        virtual_k = 'virtual/%d/observations' % (index)
        out_dataset.create_dataset(virtual_k, data=data, compression='gzip')
        #vsource = h5py.VirtualSource(out_dataset[virtual_k])
        #length = vsource.shape[0]
        #layout[offset : offset + length] = vsource
        #offset += length
        return offset

    print('Rendering images...')
    images = []
    offset = 0
    virt_idx = 0
    
    for n in tqdm.tqdm(range(N)):
        if args.adroit:
            state_dict = {k: state_datasets[k][n] for k in info_keys}
            env.set_env_state(state_dict)
            image = env.sim.render(width=args.image_size, height=args.image_size,
                                         mode='offscreen', camera_name=None, device_id=0)
            image = image[::-1,:,:]
        else:
            qpos = state_datasets['qpos'][n]
            qvel = state_datasets['qvel'][n]
            env.set_state(qpos, qvel)
            image = env.render('rgb_array', width=args.image_size, height=args.image_size)

        image = to_bytes(image, quality=args.jpg_quality)
        # TODO: resize
        images.append(image)
        if len(images) >= 10000:
            data = np.array(images)
            offset = create_virtual(virt_idx, data, offset, layout)
            virt_idx += 1
            images = []

    if len(images) > 0:
        data = np.array(images)
        print('shape:', data.shape)
        create_virtual(virt_idx, data, offset, layout)
        virt_idx += 1
    #out_dataset.create_virtual_dataset('observations', layout, fillvalue=0.0)
    out_dataset['metadata/observation_encoding'] = 'bytes_jpeg'
    out_dataset['metadata/num_chunks'] = virt_idx
    out_dataset['metadata/jpg_quality'] = args.jpg_quality

