# C2ROM
Pytorch implementation of "A Chronological and Cooperative Route Optimization Method for Heterogeneous Vehicle Routing Problem"

## Dependencies
- Python>=3.8
- NumPy
- SciPy
- Pytorch=1.12
- tqdm
- Matplotlib

## Usage
### Generate data
Before start training or test, generate validation and test data.(Training data is generated on the fly.) :
```
python make_val_dataset.py
python make_test_dataset.py
```

### Train
With single GPU(ex. V5-C80):
```
python train.py --n_custs 80 --n_agents 5
```
With multiple GPUs(ex. with 2 GPUs):
```
CUDA_VISIBLE_DEVICES=0, 1 python train.py --n_custs 80 --n_agents 5
```
For more detailed options:
```
python train.py --help
```
Training log(.csv) and checkpoint file(.pt) are output to "Results/problem name/execution time"

### Test
To test greedy strategy:
```
python test.py path/to/hoge.pt
```
To test sampling strategy(ex. with 1280 samples):
```
python test.py path/to/hoge.pt --n_sampling 1280
```
To test generalization ability(ex. with 1000 customers):
```
python test.py path/to/hoge.pt --n_custs 1000
```
