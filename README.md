# L-GCN
An implementation of Latent-Graph Convolutional Networks based on [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric), as explained in our paper:

Floris A.W. Hermsen, Peter Bloem, Fabian Jansen & Wolf B.W. Vos, [End-to-End Learning from Complex Multigraphs with Latent-Graph Convolutional Networks](https://arxiv.org/abs/1908.05365).


### Dependencies
See `requirements.txt` or `environment.yml` (conda).

### Data
The synthetic transaction networks can be found in the `/data` folder as zipped data files in pickle format. Testing can be done with files followed by a `_tiny` suffix. These contain less than 500 nodes and around 1250 transaction sets.

### Models
Model code can be found in `model.py`.
Example notebook can be found in the main directory.
