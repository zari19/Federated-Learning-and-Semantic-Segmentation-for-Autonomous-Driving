# Federated Learning and Semantic Segmentation for Autonomous Driving
### Machine Learning and Deep Learning a.a. 2022/2023
#### Politecnico di Torino

In this project we present an analysis of various sce-
narios of Semantic Segmentation and Federated Learning
implementations for autonomous driving. The analysis is
based on comparing these various approaches with a cen-
tralized baseline done with the IDDA Dataset. Moreover,
we explore the application of a novel approach, Federated
source-Free Domain Adaptation (FFreeDA), testing it with
different parameters to gain deeper insights

## Setup
#### Environment
If not working on CoLab, install environment with conda (preferred): 
```bash 
conda env create -f mldl23fl.yml
```

## How to run
The file ```MLDLD23.ipynb``` orchestrates training. The project is developed through five steps, each one can be executed by specifying ```--step ``` number in the command line, followed by all the required arguments. 

- **Example_step_2** (IDDA Dataset)
```bash
python steps.py --step 2 --dataset idda --model deeplabv3_mobilenetv2 --num_rounds 200 --num_epochs 2 --clients_per_round 8  
```
## Step 1
Centralized Approach and Data Augmentation on IDDA.

## Step 2
Supervised Federated Learning experiments trained and tested on IDDA.

## Step 3
Domain Adaptation task, pre-training phase on GTAV dataset, testing on IDDA.

## Step 4
Federated Self-training using Pseudo-Labels.

## Step 5
YOLOv8 Ensemble Learning, trained on IDDA tested on CityScapes.

## Authors
Valerio Mastrianni, Lal Akin, Riccardo Zanchetta
