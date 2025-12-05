# DFS_SE: A Powerful Framework for Image Classification on CIFAR Datasets


## Environment Requirements
- Python: 3.13 or higher
- Dependencies:
  - albumentations>=2.0.8
  - jupyter>=1.1.1
  - matplotlib>=3.10.6
  - numpy>=2.3.2
  - opencv-python>=4.11.0.86
  - pandas>=2.3.2
  - pillow>=11.3.0
  - scikit-learn>=1.7.1
  - seaborn>=0.13.2
  - torch>=2.8.0
  - torch-summary>=1.4.5
  - torchvision>=0.23.0
  - tqdm>=4.67.1


## Installation

1. Clone the Repository
```bash
git clone https://github.com/tompansir/DFSSE.git
cd DFSSE
```

2. Install Dependencies
Using uv (recommended for faster installation):
```bash
uv pip install -e .
```

Using pip:
```bash
pip install -e .
```


## Quick Start

1. Data Preparation
The framework automatically downloads and organizes CIFAR datasets:
```bash
# For CIFAR-100 (default)
python main.py --dataset cifar100 --data_dir ./data

# For CIFAR-10
python main.py --dataset cifar10 --data_dir ./data
```

2. Data Augmentation
Enable data augmentation to improve model generalization:
```bash
python main.py --dataset cifar100 --aug_count 2  # Generate 2 augmented samples per image
```

3. Model Training
Start training with default parameters:
```bash
python main.py --dataset cifar100 \
               --batch_size 64 \
               --num_epochs 60 \
               --lr 0.1 \
               --weight_decay 5e-4 \
               --output_dir ./results
```

4. Evaluation
Training automatically generates evaluation reports. View results with:
```bash
cat results/full_evaluation.txt
```


## Project Structure
```plaintext
DFS_SE/
├── main.py                  # Main pipeline entry (data + training + evaluation)
├── graphs/                  # some graphs of this research
└── scripts/
    ├── data_download.py     # CIFAR-10/100 downloaders with folder organization
    ├── data_augmentation.py # Class-specific augmentation utilities
    ├── model_architectures.py # Core model definitions (DFS_SE module)
    ├── train_utils.py       # Training loop, optimizers, and checkpointing
    └── evaluation_metrics.py # ROC curves, confusion matrices, and accuracy metrics
```


## Project Overview
DFS-SE is a high-performance deep learning framework designed for image classification tasks on CIFAR-10 and CIFAR-100 datasets. The core innovation lies in the Parallel_DFS_SE dual-branch attention mechanism, which integrates Deep Feature Search (DFS) and Squeeze-and-Excitation (SE) modules to dynamically enhance feature representation. This framework achieves state-of-the-art performance on CIFAR benchmarks while maintaining efficient computation.


## Key Features
- **Innovative Attention Mechanism**: Combines parallel SE and DFS-SE branches with learnable fusion weights to balance channel and spatial feature contributions
- **Comprehensive Pipeline**: End-to-end support for data downloading, augmentation, model training, and multi-metric evaluation
- **Strong Performance**: 73.72% Top-1 accuracy on CIFAR-100 (3.42% improvement over baseline) and 92.06% on CIFAR-10 (2.13% improvement)
- **Adaptive Data Augmentation**: Class-specific augmentation strategies (low/mid/high performance classes) to address classification imbalance
- **Self-Distillation Training**: Incorporates knowledge distillation with warm-up strategy to improve generalization
- **Reproducibility**: Configurable random seeds, detailed logging, and checkpointing for consistent experimental results


## Performance Improvement Summary

| Dataset    | Baseline Accuracy | Our Framework Accuracy | Improvement |
|------------|-------------------|------------------------|-------------|
| CIFAR-100  | 70.30%            | 73.72%                 | +3.42%      |
| CIFAR-10   | 89.93%            | 92.06%                 | +2.13%      |


## Model Performance Comparison on CIFAR-100

| Model                | Top-1 Accuracy | Top-5 Accuracy |
|----------------------|----------------|----------------|
| AlexNet              | 13.92%         | 37.56%         |
| ResNet50             | 38.61%         | 67.58%         |
| SDResNet101          | 38.96%         | 68.58%         |
| ShuffleNet-v2 (1.0×) | 42.14%         | 72.90%         |
| SE-ResNet-152        | 49.74%         | 75.31%         |
| EfficientNet-B0      | 48.70%         | 72.15%         |
| ResNet18             | 50.99%         | 76.07%         |
| MobileNet-v2         | 52.54%         | 77.45%         |
| GoogLeNet            | 54.23%         | 78.80%         |
| DenseNet-121         | 58.92%         | 82.16%         |
| **Our Framework (Ours)** | **73.72%**   | **91.94%**     |


## Model Performance Comparison on CIFAR-10

| Model                | Top-1 Accuracy | Top-5 Accuracy |
|----------------------|----------------|----------------|
| AlexNet              | 58.78%         | 94.02%         |
| SDResNet101          | 70.02%         | 97.41%         |
| SE-ResNet-152        | 71.26%         | 97.25%         |
| EfficientNet-B0      | 69.89%         | 97.38%         |
| ResNet50             | 72.43%         | 96.78%         |
| ShuffleNet-v2 (1.0×) | 74.23%         | 97.84%         |
| MobileNet-v2         | 79.99%         | 98.63%         |
| ResNet18             | 81.59%         | 97.00%         |
| GoogLeNet            | 81.52%         | 98.85%         |
| DenseNet-121         | 85.54%         | 97.61%         |
| **Our Framework (Ours)** | **92.06%**   | **98.94%**     |


## Detailed Evaluation Results

### Classification Report (CIFAR-100)

| Class               | Precision | Recall | F1-Score | Support |
|---------------------|-----------|--------|----------|---------|
| apple               | 0.95      | 0.93   | 0.94     | 100     |
| aquarium_fish       | 0.79      | 0.88   | 0.83     | 100     |
| baby                | 0.64      | 0.66   | 0.65     | 100     |
| bear                | 0.64      | 0.54   | 0.59     | 100     |
| beaver              | 0.58      | 0.61   | 0.60     | 100     |
| bed                 | 0.75      | 0.71   | 0.73     | 100     |
| bee                 | 0.73      | 0.79   | 0.76     | 100     |
| beetle              | 0.70      | 0.72   | 0.71     | 100     |
| bicycle             | 0.89      | 0.92   | 0.91     | 100     |
| bottle              | 0.84      | 0.78   | 0.81     | 100     |
| ...                 | ...       | ...    | ...      | ...     |
| wolf                | 0.79      | 0.77   | 0.78     | 100     |
| woman               | 0.52      | 0.50   | 0.51     | 100     |
| worm                | 0.68      | 0.78   | 0.73     | 100     |
| **Overall**         | **0.74**  | **0.74**| **0.74** | **10000**|

### Top-K Accuracy
- Top-1 Accuracy (Exact Match): 73.71%
- Top-5 Accuracy (Correct in Top 5): 91.94%

### Top 10 Most Common Misclassifications
1. 'boy' misclassified as 'girl': 21 times
2. 'woman' misclassified as 'girl': 20 times
3. 'oak_tree' misclassified as 'maple_tree': 20 times
4. 'maple_tree' misclassified as 'oak_tree': 15 times
5. 'boy' misclassified as 'man': 14 times
6. 'baby' misclassified as 'girl': 14 times
7. 'maple_tree' misclassified as 'willow_tree': 14 times
8. 'girl' misclassified as 'woman': 13 times
9. 'bed' misclassified as 'couch': 13 times
10. 'snake' misclassified as 'worm': 12 times


## Ablation Study (CIFAR-100)

| Model Configuration                          | Top-1 Accuracy | Improvement over Baseline |
|-----------------------------------------------|----------------|---------------------------|
| Baseline                                      | 70.30%         | -                         |
| Baseline + PromptLite                         | 69.62%         | -0.68%                    |
| Baseline + DFS-SE (Series-Parallel)           | 72.39%         | +2.09%                    |
| Baseline + Dropout                            | 72.07%         | +1.77%                    |
| Baseline + Prompt + Dropout                   | 71.61%         | +1.31%                    |
| Baseline + Prompt + DFS-SE                    | 72.69%         | +2.39%                    |
| Baseline + DFS-SE + Dropout                   | 73.36%         | +3.06%                    |
| Baseline + Dropout + PromptLite + SE          | 73.38%         | +3.08%                    |
| Baseline + Dropout + PromptLite + SE-DFS (Only Series) | 72.74% | +2.44%    |
| **Complete Framework (Dropout + PromptLite + DFS-SE)** | **73.72%** | **+3.42%** |


## Key Components

1. **DFS_SE Module**: A dual-path feature enhancement module combining:
   - **SE Sub-module**: First performs channel-wise coarse screening via global average pooling and sigmoid gating
   - **DFS Sub-module**: Conducts spatial fine-grained search on SE-processed features through T-step graph traversal
   - **Fixed-weight Fusion**: Combines SE and DFS attention maps (default 50% each) to weight original features

2. **Adaptive Data Augmentation**: 
   - Class-specific strategies based on performance (Low/Mid/High)
   - Stronger augmentation for low-performance classes (e.g., 'boy', 'woman') with rotations, affine transformations, and dropout
   - Moderate augmentation for mid-performance classes
   - Light augmentation for high-performance classes (e.g., 'apple', 'bicycle')

3. **Self-Distillation Training**:
   - Knowledge distillation with temperature scaling (T=3.0)
   - Warm-up strategy (3 epochs) before introducing distillation loss
   - Gradual increase of distillation weight (α) to 0.1 for stable training

4. **End-to-End Pipeline**: Integrated data downloading, organization, augmentation, training, and evaluation with detailed logging


## Conclusion
The DFS_SE framework achieves state-of-the-art performance on CIFAR datasets through the innovative combination of DFS and SE attention mechanisms, along with effective data augmentation and training strategies. The ablation study confirms the effectiveness of each component, with the parallel DFS-SE structure and Dropout combination contributing the most to performance improvement.


## Citation
If you use this framework in your research, please cite:
```plaintext
@misc{dfs-se2024,
  title={DFS-SE: Dual-Branch Attention Framework for CIFAR Classification},
  author={Enze Pan},
  year={2026},
  url={https://github.com/tompansir/DFSSE.git}
}
```
