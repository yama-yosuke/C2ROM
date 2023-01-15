import os
import torch

SEED = 456
SAMPLE_NUM = 10240
MAX_DEMAND = 9
N_CUSTS_LIST = [20, 40, 60, 80, 100, 120, 140, 160]

def make_dataset(n_custs):
    title = 'C{}-MD{}-S{}-seed{}'.format(n_custs, MAX_DEMAND, SAMPLE_NUM, SEED)
    save_path = os.path.join("dataset", "val", "{}.pt".format(title))
    
    torch.manual_seed(SEED)
    settings = {
        "sample_num": SAMPLE_NUM,
        "n_custs": n_custs,
        "max_demand": MAX_DEMAND,
    }

    init_demand = (torch.FloatTensor(SAMPLE_NUM, n_custs + 1).uniform_(0, MAX_DEMAND).int() + 1).float()
    init_demand[:, 0] = 0  # depotのdemandは0
    location = torch.FloatTensor(SAMPLE_NUM, n_custs + 1, 2).uniform_(0, 1)
    save_data = {}
    save_data["settings"] = settings
    save_data["init_demand"] = init_demand
    save_data["location"] = location

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_data, save_path)

if __name__ == "__main__":
    for n_cust in N_CUSTS_LIST:
        make_dataset(n_cust)
