# Module 2: Softmax Regression and Classification

Welcome to our second core module! If Logistic Regression was the essential step-stone for binary classification (two classes), **Softmax** is the crucial generalization that allows us to handle real-world problems involving *multiple* distinct categories. This module teaches you how models interpret scores in a multi-class setting.

## Learning Objectives
After completing this lesson, you will be able to:

1.  **Explain Softmax's Role:** Understand why and when Softmax is necessary for multi-class classification (i.e., when the number of classes $C > 2$).
2.  **Model Structure:** Describe how a Softmax model must calculate separate linear scores for every single class, resulting in multiple weight vectors and biases.
3.  **Prediction Mechanics:** Master the two key steps: calculating raw logits ($\mathbf{z}$) and then using the `argmax` function to select the final predicted class label.

---

## Part I: The Need for Multi-Class Classification
In real-world AI applications—such as image classification, spam detection, or recommendation systems—our model must handle many possible outcomes simultaneously. These are called **multi-class problems**.

When dealing with multiple classes, a model needs more than one single line (as in binary logistic regression). It requires an entire system of linear equations—one set of weights and biases for *each* class—to score the input against every possibility.

### The Softmax Model Structure
Softmax fundamentally uses **$C$ separate linear classification lines**, where $C$ is the total number of classes. For a given input feature vector $\mathbf{x}$ with $F$ features, the model calculates $C$ different raw scores, or **logits** ($\mathbf{z}$):

$$\begin{cases} z_0 = \mathbf{w}_0 \cdot \mathbf{x} + b_0 & (\text{Score for Class } 0) \\ z_1 = \mathbf{w}_1 \cdot \mathbf{x} + b_1 & (\text{Score for Class } 1) \\ \vdots & \\ z_{C-1} = \mathbf{w}_{C-1} \cdot \mathbf{x} + b_{C-1} & (\text{Score for Class } C-1) \end{cases}$$

*   **Weight Matrix ($\mathbf{W}$):** The model’s weights are no longer a single vector, but a matrix of size $C \times F$. Each row of $\mathbf{W}$ corresponds to the full set of weights $\{\mathbf{w}_i\}$ for one specific class.
*   **Bias Vector ($\mathbf{b}$):** There is an independent bias term $b_i$ for every class.

The model's forward pass computes this entire vector of raw scores: $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

---

## Part II: From Logits to Probabilities
The output $\mathbf{z}$ is a tensor of $C$ logits. These logits are useful for training (as they flow directly into the Cross-Entropy Loss function), but they must be converted into normalized probabilities ($\hat{\mathbf{p}}$) for human interpretation and final decision making. This is where the Softmax formula operates:

$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

This function ensures two critical properties:
1.  **Positive:** Every resulting probability $\hat{p}_i$ is positive (since $e^x > 0$).
2.  **Normalization:** The sum of all probabilities across all classes must equal 1 ($\sum \hat{p}_i = 1$).

---

## Part III: Making the Final Prediction

Softmax provides a probability distribution, but we still need a single integer class label for classification. This is done using the `argmax` function.

### The Role of Argmax
The **Argmax** function takes the full vector of probabilities $\hat{\mathbf{p}}$ and simply returns the *index* corresponding to the highest probability value.

$$\text{Predicted Class} = \text{argmax}(\hat{p}_0, \hat{p}_1, \dots, \hat{p}_{C-1})$$

**Example:** If Softmax outputs $[0.15, 0.80, 0.05]$, the `argmax` function returns $\mathbf{1}$. This means the model predicts Class 1 is the most likely outcome.

### Implementation in PyTorch
The prediction process involves three distinct steps:

1.  **Linear Pass:** Compute raw logits ($\mathbf{z}$) using the linear layers (via $\mathbf{W}\mathbf{x} + \mathbf{b}$).
2.  **Probability Conversion:** Apply `torch.softmax(z, dim=1)` to get the probability distribution ($\hat{\mathbf{p}}$).
3.  **Prediction Retrieval:** Apply `torch.argmax()` along dimension 1 (`dim=1`) to convert probabilities back into the final class index ($\text{y}_{\text{hat}}$).

This structured approach—Linear $\to$ Softmax $\to$ Argmax—is the backbone of nearly every multi-class deep learning model.