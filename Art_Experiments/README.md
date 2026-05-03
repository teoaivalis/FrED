# FrED: Art Experiments (Artbench)

This folder contains the implementation and evaluation suite for **FrED** (Framework for External and Domain-Aware Influence Analysis) applied to the domain of abstract artistic image synthesis. 

Our experimental framework is built upon the [D-TRAK](https://github.com/sail-sg/D-TRAK) codebase. We extend the baseline by incorporating domain-specific Knowledge Graphs to enable high-fidelity, black-box training data attribution.

---

## 📂 Folder Structure

*   **`D-TRAK/Artbench2/`**: Experiments and scripts for a 2-class subset of the Artbench dataset.
*   **`D-TRAK/Artbench5/`**: Experiments and scripts for a 5-class subset of the Artbench dataset.
*   **`Art_KG_Data/`**: All the necessary data for the domain knowledge of the Art Domain.
*   **`/../../methods/fred_attribution/`**: Located within the `methods` directory of both Artbench folders, this contains the core implementation of our proposed FrED framework.

---

## 🚀 How to Run



## Quickstart
Check [quickstart.ipynb](quickstart.ipynb) to conduct data attribution on pre-trained diffusion models loaded from huggingface directly!

### Replicating the Paper's Results

> The experimental pipeline and evaluation metrics for the artistic domain are based on the methodology established in the [D-TRAK repository](https://github.com/sail-sg/D-TRAK). We utilize their benchmarking suite to ensure a fair and standardized comparison between FrED and existing gradient-based estimators.

### Setup
To get started, follow these steps:

1. **Clone the GitHub Repository:** Begin by cloning the repository using the command:
   ```shell
   git clone https://github.com/sail-sg/D-TRAK.git
   ```
2. **Set Up Python Environment:** Ensure you have a version 3.8.
   name:
   ```shell
   conda create -n dtrak python=3.8 -y
   conda activate dtrak
   ```
3. **Install Dependencies:** Install the necessary dependencies by running:
   ```shell
   pip install -r requirements.txt
   ```

### Commands for LDS evaluation
We provide the commands to run experiments on CIFAR-2. 
It is easy to transfer to other datasets.

1. **Data pre-processing:** 
    ```shell
    cd CIFAR2
    ```
    Run [00_EDA.ipynb](CIFAR2/00_EDA.ipynb) to create dataset splits and subsets of the training set.

4. **Train a diffusion model and generate images:** 
    ```shell
    bash scripts/run_train.sh 0 18888 5000-0.5
    bash scripts/run_gen.sh 0 0 5000-0.5
    ```
5. **Construct the LDS benchmark:** 
    
    Train 64 models corresponding to 64 subsets of the training set
    ```shell
    bash scripts/run_lds_val_sub.sh 0 18888 5000-0.5 0 63
    ```
    Evaluate the model outputs on the validation set
    ```shell
    bash scripts/run_eval_lds_val_sub.sh 0 0 5000-0.5 idx_val.pkl 0 63
    bash scripts/run_eval_lds_val_sub.sh 0 1 5000-0.5 idx_val.pkl 0 63
    bash scripts/run_eval_lds_val_sub.sh 0 2 5000-0.5 idx_val.pkl 0 63
    ```
    Evaluate the model outputs on the generation set
    ```shell
    bash scripts/run_eval_lds_val_sub.sh 0 0 5000-0.5 idx_gen.pkl 0 63
    bash scripts/run_eval_lds_val_sub.sh 0 1 5000-0.5 idx_gen.pkl 0 63
    bash scripts/run_eval_lds_val_sub.sh 0 2 5000-0.5 idx_gen.pkl 0 63
    ```
6. **Compute gradients:** 

    We shard the training set into 5 parts, each has 1000 examples.

    Use the following commands to compute the gradients to be used for TRAK. 

    ```shell
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 0 ddpm/checkpoint-8000 loss uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 1 ddpm/checkpoint-8000 loss uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 2 ddpm/checkpoint-8000 loss uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 3 ddpm/checkpoint-8000 loss uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 4 ddpm/checkpoint-8000 loss uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-val.pkl 0 ddpm/checkpoint-8000 loss uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-gen.pkl 0 ddpm/checkpoint-8000 loss uniform 10 32768
    ```

    Use the following commands to compute the gradients to be used for D-TRAK. 

    ```shell
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 0 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 1 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 2 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 3 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-train.pkl 4 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-val.pkl 0 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    bash scripts/run_grad.sh 0 0 5000-0.5 idx-gen.pkl 0 ddpm/checkpoint-8000 mean-squared-l2-norm uniform 10 32768
    ```

7. **Compute the TRAK/D-TRAK attributions and evaluate the LDS scores**

    Run notebooks in [methods/04_if](CIFAR2/methods/04_if).

    The implementations of other baselines can also be found in [methods](CIFAR2/methods).

### Commands for counterfactual evaluation

1. **Data pre-processing**

    Run this [notebook](CIFAR2/methods/04_if/get_indices_gen.ipynb) first to get the indices of those training examples to be removed.

2. **Retrain models after removing the top-influenctial training examples**
    ```shell
    bash scripts/run_counter.sh 0 18888 5000-0.5 0 59
    ```

3. **Generate images using the retrained models**

    Run [02_counter.ipynb](CIFAR2/02_counter.ipynb)

4. **Measure l2 distance**

    Run [03_counter_eval_l2.ipynb](CIFAR2/03_counter_eval_l2.ipynb)

5. **Measure CLIP cosine similarity**

    Run [03_counter_eval_clip.ipynb](CIFAR2/03_counter_eval_clip.ipynb)