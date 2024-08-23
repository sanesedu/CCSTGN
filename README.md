# Channel-Centric Spatio-Temporal Graph Networks for Network-based Intrusion Detection

## Abstract 

The increasing frequency and complexity of cyberattacks against critical digital infrastructures require novel methods that can detect intrusions in a timely manner. Recent work has explored the use of Graph Neural Networks (GNNs) for network-based intrusion detection, adapting network traffic flow data to traditional GNN representations. In this work, we propose an alternative approach based on a novel combination of a graph representation of network traffic that represents communication channels as nodes and of a continuous temporal representation. 
Our proposed architecture, called Channel-Centric Spatio-Temporal Graph Networks (CCSTGN), can be trained using different learning strategies and is introduced together with a detailed data preprocessing strategy.
We present an experimental evaluation of our proposed CCSTGN architecture using different learning strategies, from which we conclude that our proposal is able to outperform multiple existing GNN-based methods in terms of various classification metrics and that the data preprocessing procedure can be of significant importance for the performance of the models.

## 0) Conda environment

`conda env create -f environment.yml`

## 1) Data preparation

1. Create `./datasets` directory
2. Download and unzip NF-UNSW-NB15-v2 dataset inside that directory ([link](https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848))

## 2) Data preprocessing

1. `python ./data_processing/data_preprocessing.py --task 'mapping'`
2. `python ./data_processing/data_preprocessing.py --task 'prepare'`
3. `python ./data_processing/data_preprocessing.py --task 'encode'`

## 3) Experiment execution

Execute `experiment1.py` or `experiment2.py` for each model according to their corresponding arguments. For instance, for the validation run of CCSTGN-iso:

```
python experiment1.py --experiment 'validation' --variant 'iso' \
                --bs 10000 \
                --n_layers 1 \
                --n_neighbors 5 \
                --agg_function 'mean' \
                --memory_dim 20 \
                --n_epochs 7
```
