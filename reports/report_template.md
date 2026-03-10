\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{parskip}

\begin{document}

\begin{center}
    {\Large DSC 140B Final Project Report}\\[4pt]
    SoCalGuessr Model Report
\end{center}

\section*{\textbf{Human Baseline}}
I evaluated myself on the SoCalGuessr human baseline game at \texttt{https://eldridgejm.github.io/SoCalGuessr/}. My final score was \textbf{19/50 = 38\%}. Figure~\ref{fig:human-baseline} shows the confusion matrix screenshot from that run. The human baseline is much lower than the model validation accuracy, which suggests that the learned visual features are capturing location cues that are difficult to identify consistently by hand.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.72\linewidth]{human_baseline.png}
    \caption{Human baseline result and confusion matrix screenshot from the SoCalGuessr website.}
    \label{fig:human-baseline}
\end{figure}

\section*{\textbf{Final Model Architecture}}
My final model is a \textbf{pretrained EfficientNet-B0} implemented with \texttt{torchvision}. I used the ImageNet-pretrained EfficientNet-B0 backbone and replaced the final classifier so that the model outputs logits for the six city classes: Anaheim, Bakersfield, Los Angeles, Riverside, SLO, and San Diego. In code, the final classifier is changed from the default 1000-way ImageNet classifier to a new linear layer with input dimension 1280 and output dimension 6.

For the task-specific head, the model has \textbf{no additional hidden fully connected layers}. The custom classifier is a single linear layer mapping the 1280-dimensional EfficientNet feature vector to 6 output logits. Thus, the task-specific head has:
\begin{itemize}[leftmargin=1.2em]
    \item \textbf{Hidden layers:} 0 additional hidden fully connected layers
    \item \textbf{Final linear layer:} 1280 \(\rightarrow\) 6
\end{itemize}

The total number of parameters in the final model is approximately 4.02 million parameters. This comes from the standard torchvision EfficientNet-B0 parameter count with the original 1000-class classifier replaced by a 6-class linear layer. All parameters were trainable, since I did not enable backbone freezing in the training command.

The activations used in the model are the standard \textbf{SiLU (Swish)} nonlinearities inside the EfficientNet-B0 backbone. The final classifier layer is linear and does not apply a separate output activation. During training, the loss function is cross-entropy, which applies the softmax normalization implicitly when computing the loss.

\section*{\textbf{Training Procedure}}
I trained the model using the provided training images, where the class label is extracted from the filename prefix before the first hyphen. I used an \textbf{80/20 stratified train/validation split} with random seed 42 so that each city remains represented in both partitions. The final training run used the following command:

\begin{center}
\texttt{python train.py --data-dir data --arch efficientnet\_b0 --epochs 12 --batch-size 64 --lr 1e-4}
\end{center}

The model was trained with Adam and cross-entropy loss for 12 epochs using a learning rate of \(10^{-4}\), weight decay \(10^{-4}\), batch size 64, and image size \(224 \times 224\). Training was run on CPU.

For preprocessing, all images were resized to \(224 \times 224\). During training, I applied random horizontal flipping and color jitter, followed by ImageNet normalization. During validation and inference, I used resizing plus ImageNet normalization only. The model checkpoint with the highest validation accuracy was saved automatically during training.

The best validation accuracy achieved during training was \textbf{0.9200} (92.00\%). The validation accuracy improved steadily through the middle epochs and then leveled off, while the training loss continued decreasing, which suggests mild late-stage overfitting. The best checkpoint was therefore selected based on validation accuracy rather than simply using the final epoch.

\textbf{Training time:} The training code was run on CPU and took approximately 8 hours 49 minutes of wall-clock time (8:48:56.94 total).

Figure~\ref{fig:training-curve} shows the training curve. In the current training code, the curve is recorded as \textbf{iteration number versus empirical risk}, where empirical risk is the mini-batch cross-entropy loss collected during training.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{training_curve.png}
    \caption{Training curve showing iteration number versus empirical risk (mini-batch training loss).}
    \label{fig:training-curve}
\end{figure}

In summary, the final system uses a pretrained EfficientNet-B0 backbone with a 6-class linear classifier, trained with Adam and cross-entropy loss on a stratified train/validation split. The final validation result of 92.00\% suggests that the model learned strong visual cues for distinguishing Southern California cities.

\end{document}
