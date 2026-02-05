# Zero-Flow Encoders

The implementation of [Zero-Flow Encoders](https://arxiv.org/abs/2602.00797)

## Abstract
Flow-based methods have achieved significant success in various generative modeling tasks, capturing nuanced details within complex data distributions. However, few existing works have exploited this unique capability to resolve fine-grained structural details beyond generation tasks. This paper presents a flow-inspired framework for representation learning. First, we demonstrate that a rectified flow trained using independent coupling is zero everywhere at $t = 0.5$ if and only if the source and target distributions are identical. We term this property the **zero-flow criterion**. Second, we show that this criterion can certify conditional independence, thereby extracting **sufficient information** from the data. Third, we translate this criterion into a tractable, simulation-free loss function that enables learning amortized Markov blankets in graphical models and latent representations in self-supervised learning tasks. Experiments on both simulated and real-world datasets demonstrate the effectiveness of our approach.

## Prerequisites

Before you begin, ensure you have met the following requirements:
* You have installed the latest version of [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* You have a `Windows/Linux/Mac` machine.

## Installation

To install the necessary packages and set up the environment, follow these steps:

### Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/probabilityFLOW/zfe.git
cd zfe
```

### Create the Conda Environment

To create the Conda environment with all the required dependencies, run:

```bash
conda env create -f environment.yaml
```

## Usage


### Training
After activating the environment, you can run the scripts or use the modules provided in the repository. 
```bash
conda activate zfe
```

Learning Markov Blanket on Graphical Data including Gaussian Nonparanormal and Truncated Gaussian random variable:
```bash
python ToyLattice.py
```

PatchedCIFAR10 Zero-Flow

```bash
python PatchedCIFAR10.py
```

ColoredMNIST Zero-Flow
```bash
python ColoredMNIST.py
```

ColoredMNIST SimCLR
```bash
python ColoredMNISTSimCLR.py
```

PatchedCIFAR10 SimCLR
```bash
python PatchedCIFARSimCLR.py
```

To reproduce the MAE experiment result in shortcut problem:
```bash
cd ./mae_shortcut/
bash script.sh
```
Then you can run tasks (e.g. `lin_probe.ipynb`) in the folder.