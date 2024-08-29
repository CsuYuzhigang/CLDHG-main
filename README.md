# CLDHG: Contrastive Learning on Dynamic Heterogeneous Graphs
## Code structure
| *folder* |              description               |
|:--------:|:--------------------------------------:|
|   data   |               Datasets.                |
|  CLDHG   | CLDHG implementation code is provided. |

## Datasets

|   **Dataset**    | # nodes | # temporal edges | # node types | # edge types | # snapshots |
|:----------------:|:-------:|:----------------:|:------------:|:------------:|:-----------:|
|   **Twitter**    | 100,000 |      63,410      |      1       |      3       |      7      |
| **MathOverflow** | 24,818  |     506,550      |      1       |      3       |     11      |
|    **EComm**     | 37,724  |      91,033      |      2       |      4       |     11      |


## Usage
```python
# Twitter
python main.py --dataset Twitter --hidden_dim 128 --output_dim 64 --n_layers 2 --fanout 20,20 --snapshots 7 --views 4 --strategy random --epochs 200 --GPU 0

# MathOverflow
python main.py --dataset MathOverflow --hidden_dim 128 --output_dim 64 --n_layers 2 --fanout 20,20 --snapshots 11 --views 3 --strategy random --epochs 200 --GPU 0

# EComm
python main.py --dataset EComm --hidden_dim 128 --output_dim 64 --n_layers 2 --fanout 20,20 --snapshots 11 --views 4 --strategy random --epochs 200 --GPU 0

```

## Dependencies

- Python 3.7
- PyTorch 1.9.0+cu111
- dgl-cu110 0.7.1

## Reference
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{
  title={CLDHG: Contrastive Learning on Dynamic Heterogeneous Graphs},
  author={},
  booktitle={},
  pages={},
  year={},
  organization={}
}
```