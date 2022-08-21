from torch.utils.data import Dataset
import torch
import os
import pickle
from .state import StateTSP
# from .beam_search import beam_search


class TSP(object):
    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    # @staticmethod
    # def beam_search(input, fixed_embedding, beam_size, expand_size=None,
    #                 compress_mask=False, model=None, max_calc_batch_size=100000):
    #     assert model is not None, "Provide model"

    #     def propose_expansions(used_for_compute, beam):
    #         return model.propose_expansions(used_for_compute, fixed_embedding,
    #             beam, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
    #         )

    #     state = TSP.make_state(
    #         input, visited_dtype=torch.int64 if compress_mask else torch.uint8
    #     )

    #     return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=10000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:

            assert os.path.splitext(filename)[1] == '.pkl'
            with open(filename, 'rb') as f:
                # data_all is dict type: 'data': coords, 'solution': solution of corresp. instances,
                # 'extra_info': [array of optimal sequence order, array of k-nearest node in this optimal order]
                data_all = pickle.load(f)
                data = data_all['data']
                soln = data_all['seq']
                val = data_all['val']
                time = data_all['time']
                # extra_info = data_all['extra_info']
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
                self.soln = [torch.LongTensor(row) for row in (soln[offset:offset + num_samples])]
                self.opt_val = torch.FloatTensor(val[offset:offset + num_samples])
                self.time = torch.FloatTensor(time[offset:offset + num_samples])
                # self.opt_order = [torch.LongTensor(row[1]).squeeze() for row in (extra_info[offset:offset + num_samples])]

        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            return self.data[idx], self.soln[idx], self.opt_val[idx], self.time[idx]

        except:
            return self.data[idx]
