# DSC 140B Final Project Report

## **Human Baseline**

- Human accuracy: XX.X%
- Brief note on how difficult the task felt.
- Insert screenshot of SoCalGuessr confusion matrix here.

![Human confusion matrix](../outputs/human_confusion_matrix.png)

## **Final Model Architecture**

- Final architecture:
- Number of hidden layers:
- Units / channels per hidden layer:
- Activations used:
- Number of parameters:
- Input image size:
- Whether pretrained weights were used:

Example writeup:

> My final model was a fine-tuned EfficientNet-B0 convolutional neural network pretrained on ImageNet. The feature extractor consists of multiple convolutional blocks with nonlinear activations and squeeze-and-excitation layers, followed by a final linear classification head with 6 output units, one for each city class. The final layer uses a softmax implicitly through the cross-entropy loss. The model contains approximately ______ trainable parameters.

## **Training Procedure**

- Optimizer:
- Loss function:
- Batch size:
- Learning rate:
- Number of epochs:
- Training time:
- Validation accuracy:
- Any augmentations used:

Include the required training curve below.

![Training curve](../outputs/training_curve.png)

Additional notes:
- Describe how you split train vs validation.
- Mention early stopping, checkpoint selection, and any tuning.
