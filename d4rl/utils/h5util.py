import h5py

def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys

def contains_key(h5file, key):
    return key in get_keys(hfile)

