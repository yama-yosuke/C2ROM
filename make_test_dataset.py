import os

import numpy as np
import torch

"""
Generate the same test data used in https://github.com/Demon0312/HCVRP_DRL
"""
SEED = 24610  # the last seed used for generating HCVRP data
SAMPLE_NUM = 1280
MAX_DEMAND = 9
N_CUSTS_LIST = [20, 40, 60, 80, 100, 120, 140, 160]


def make_dataset(n_custs):
    np.random.seed(SEED)
    save_path = os.path.join(
        "dataset", "test", "C{}-MD{}-S{}-seed{}.pt".format(n_custs, MAX_DEMAND, SAMPLE_NUM, "=TEST")
    )

    settings = {
        "sample_num": SAMPLE_NUM,
        "n_custs": n_custs,
        "max_demand": MAX_DEMAND,
    }

    locations = []
    init_demands = []
    dataset_size = int(SAMPLE_NUM / 10)
    for seed in range(24601, 24611):
        rnd = np.random.RandomState(seed)
        location_ = rnd.uniform(0, 1, size=(dataset_size, n_custs + 1, 2))
        demand_ = rnd.randint(1, 10, [dataset_size, n_custs + 1])
        # for compatibility with Li(index of depot is 0 in this code)
        location = np.concatenate([np.expand_dims(location_[:, -1, :], 1), location_[:, :-1, :]], axis=1)
        demand = np.concatenate([np.zeros((dataset_size, 1)), demand_[:, :-1]], axis=1)
        locations.append(torch.from_numpy(location))
        init_demands.append(torch.from_numpy(demand))

    save_data = {}
    save_data["settings"] = settings
    save_data["init_demand"] = torch.cat(init_demands, dim=0).float()
    save_data["location"] = torch.cat(locations, dim=0).float()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_data, save_path)


if __name__ == "__main__":
    for n_cust in N_CUSTS_LIST:
        make_dataset(n_cust)
