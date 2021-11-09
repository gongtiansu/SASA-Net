# SASA-Net
## About The Project

The implementation of the paper "SASA-Net: A spatial-aware self-attention mechanism for building protein 3D structure directly from inter-residue distances".

## Getting Started
### Prerequisites
Install [PyTorch 1.6+](https://pytorch.org/),
[python
3.7+](https://www.python.org/downloads/)

### Installation

1. Clone the repo
```sh
git clone https://github.com/gongtiansu/SASA-Net.git
```

2. Install python packages
```sh
cd SASA-Net
pip install -r requirements.txt
```

## Usage
1. Generate estimated inter-residue distance using [ProFOLD](https://github.com/fusong-ju/ProFOLD)
2. Run SASA-Net
```sh
run_SA.sh --fasta <fasta> --feat <profold_npz> --output <output_pdb>
```

## Example
```sh
cd example
./run_example.sh
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
