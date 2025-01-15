# BOVIS: Bias-Mitigated Object-Enhanced Visual Emotion Analysis

## Abstract
Visual emotion analysis is an emerging field that aims to predict emotional responses elicited by visual stimuli. While recent advances in deep learning have significantly improved emotion detection capabilities, existing methods often fall short by focusing exclusively on either holistic visual features or semantic content, neglecting their interplay. To address this limitation, we introduce **BOVIS**, a **Bias-Mitigated Object-Enhanced Visual Emotion Analysis** framework. 

In order to capture subtle relationships between visual and semantic features and enrich the understanding of emotional contexts, **BOVIS** leverages pre-trained models to extract comprehensive image features, integrate object-level semantics, and enhance contextual information. Additionally, **BOVIS** incorporates a bias mitigation strategy, which includes an adjusted **Mean Absolute Error** loss function combined with an **Inverse Probability Weighting** approach, to address dataset imbalances and improve fairness in emotion prediction.

Extensive evaluations on multiple benchmark datasets validate the effectiveness of the BOVIS framework in advancing visual emotion analysis. The results demonstrate that the synergy between object-specific features and holistic visual representations enhances the accuracy and interpretability of emotion analysis, while the bias mitigation optimization improves fairness and increases reliability.

## Model Architecture

![Model Architecture](BOVIS_Framework.png)

The above image represents the architecture of **BOVIS**, showcasing how the framework integrates visual features, object-level semantics, and bias mitigation strategies.

## Key Features
- **BOVIS** leverages **pre-trained models** to extract comprehensive image features, integrate object-level semantics, and enhance emotional context understanding.
- It incorporates a **Bias Mitigation** strategy to address dataset imbalances, improving fairness and reliability in emotion prediction.
- **Evaluation**: Extensive tests on multiple benchmark datasets validate the effectiveness of BOVIS, demonstrating that the synergy between object-specific features and holistic visual representations improves the accuracy and interpretability of emotion analysis, while bias mitigation optimization enhances fairness and reliability.

## Key Techniques
- **Visual Features**: Extracts diverse image features using pre-trained models.
- **Object-Level Semantics**: Integrates object-specific semantics into the analysis.
- **Bias Mitigation**: Utilizes **Mean Absolute Error Loss** and **Inverse Probability Weighting** to address dataset imbalances.

## Installation
```bash
# Clone the repository
git clone https://github.com/leeyubin10/BOVIS.git

# Install dependencies
pip install -r requirements.txt
