# Dependencies
- Python>=3.8
- NumPy
- SciPy
- Pytorch=1.12
- tqdm
- Matplotlib
- wandb(optional)

# Usage
## Generating data
Training data is generated on the fly. Generate validation and test data:
```
python make_val_dataset.py
python make_test_dataset.py
```

## Training
With single GPU:
```
python train.py --n_custs 40 --n_agents 5
```
With multi GPUs(When limiting to 2 GPUs):
```
CUDA_VISIBLE_DEVICES=0, 1 python train.py --n_custs 40 --n_agents 5
```
See options.py for more detailed option.

## Test
To get the best model of all epochs:
```
python test_all.py path/to/checkpoints
```
To test specified model(greedy):
```
python test.py path/to/hoge.pt
```
To test specified model(sampling 1280):
```
python test.py path/to/hoge.pt --n_sampling 1280
```
To test specified model on all customers(greedy):
```
python test.py path/to/hoge.pt --all_cust
```
If Memory allocation error occurs, decrease the value of `--batch_size`  
See test.py for detailed option.
