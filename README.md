# BOVIS: Bias-Mitigated Object-Enhanced Visual Emotion Analysis (CIKM'25)

## Abstract
Visual emotion analysis is a promising field that aims to predict emotional responses elicited by visual stimuli. While recent advances in deep learning have significantly improved emotion detection capabilities, existing methods often fall short due to their exclusive focus on either holistic visual features or semantic content, thereby neglecting their interplay. To address this limitation, we introduce BOVIS, a Bias-Mitigated Object-Enhanced Visual Emotion Analysis framework.
To capture the subtle relationships between visual and semantic features and enrich the understanding of emotional contexts, BOVIS leverages pre-trained models to extract comprehensive image features, integrate object-level semantics, and enhance contextual information.
Moreover, BOVIS incorporates a bias mitigation strategy that involves an adjusted Mean Absolute Error loss function alongside an Inverse Probability Weighting method to tackle dataset imbalances and enhance fairness in emotion prediction.
Comprehensive evaluations across various benchmark datasets demonstrate the effectiveness of the BOVIS framework in enhancing visual emotion analysis. The results reveal that the synergy between object-specific features and holistic visual representations improves the accuracy and interpretability of emotion analysis, while optimizing bias mitigation enhances fairness and increases reliability.

## Overall Framework

![Model Architecture](BOVIS_Framework.png)

## Installation
```bash
# Clone the repository
git clone https://github.com/leeyubin10/BOVIS.git

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### Download the Dataset
Download the dataset and split it into **train**, **validation**, and **test** sets.

### Setup the Object Detection Folder
The object detection functionality in **BOVIS** uses a pre-trained **Faster R-CNN** model, which is pretrained on the **Visual Genome** dataset.  
Set up the `object_detection` folder as described in the instructions of the repository [Faster-R-CNN-with-model-pretrained-on-Visual-Genome](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome.git) to ensure compatibility with this framework.

## Training the Model

After setting up the dataset and object detection folder, you can proceed to train the model using the configuration specified in the `config/train_config.yaml` file.

```bash
# Train the model
python train.py
```

