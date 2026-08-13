# Module 3: Multiclass Neural Networks and Softmax in Action

Welcome! This module is where all previous concepts culminate, teaching you how to build and train a fully functional **multi-class classifier**. We move beyond binary choice (0 or 1) to tackling the vast complexity of selecting from $C$ possible categories.

## Learning Objectives
By the end of this session, you will be able to:

1.  **Understand Output Structure:** Design a network output layer that correctly maps inputs to one score per class.
2.  **Differentiate Loss Functions:** Understand why `nn.CrossEntropyLoss()` is the standard loss function for multi-class problems in PyTorch and how it differs from binary losses.
3.  **Implement End-to-End Pipeline:** Successfully implement, train, and evaluate a full multiclass model on a dataset like MNIST.

---

## Part I: The Multi-Class Architecture
### 1. Output Layer Design (The Number of Neurons)
For $C$ classes, the output layer must have exactly $C$ neurons. Each neuron acts as a specialized scoring mechanism for one class.

*   **Old Way (Binary):** One linear layer mapping $\mathbf{x}$ to $z$. Single sigmoid output (1 value).
*   **New Way (Multiclass):** The final linear layer maps the hidden activations ($\mathbf{a}$) to $\mathbf{z}$, which is a vector of $C$ scores.

$$\text{Output Layer}: \mathbf{z} = \mathbf{W}_{\text{output}} \cdot \mathbf{a} + \mathbf{b}$$
*   The weight matrix $\mathbf{W}_{\text{output}}$ must have dimensions of $C \times H$ (Classes $\times$ Hidden Neurons). This means each row of the output weight matrix is a dedicated linear combination, calculating one class score independently.

### 2. Prediction: The Argmax Rule
After obtaining the logit vector $\mathbf{z}$ (the scores for all classes), we don't need to calculate probabilities manually first. We simply find the index of the highest value.
$$\text{Predicted Class} = \text{argmax}(z_0, z_1, \dots, z_{C-1})$$

---

## Part II: The PyTorch Implementation Details

### 1. Code Structure Modifications
The modifications to a binary model are minor but critical:
*   **Output Dimension:** When defining the `nn.Linear` layer for the final output, the `out_size` must be set to $C$ (the number of classes).
*   **Loss Function:** We switch from `nn.BCELoss()` to **`nn.CrossEntropyLoss()`**. This is a major simplification! It signals to PyTorch that you are performing multi-class classification, and PyTorch handles the necessary Softmax conversion internally, saving us from manual calculation and potential stability errors.
*   **Output Activation:** Crucially, we **remove the final `torch.sigmoid` activation function** in the forward pass when using `nn.CrossEntropyLoss()`. This is because the loss criterion expects raw logits ($\mathbf{z}$) and will internally apply the Softmax transformation for its calculation.

### 2. The Training Pipeline (End-to-End)
The training loop follows a robust pattern:

1.  **Data:** Load batches of $\mathbf{x}$ and true labels $\mathbf{y}$ (must be `LongTensor` with shape $N$, not $N \times 1$).
2.  **Forward Pass:** Compute the logits $\mathbf{z} = \text{model}(\mathbf{x})$.
3.  **Loss:** Calculate loss using $\text{criterion}(\mathbf{z}, \mathbf{y})$.
4.  **Optimization:** Run `zero_grad()`, `backward()`, and `step()` to adjust the weights based on the multi-class cross-entropy signal.

### 3. Evaluation Metrics
Accuracy calculation remains the same:
$$\text{Accuracy} = \frac{\sum_{i=1}^{N_{\text{test}}} (\mathbf{y}_{\text{hat}, i} == \mathbf{y}_{\text{true}, i})}{N_{\text{test}}}$$

---

## Conclusion: A Unified Deep Learning Framework
This module solidifies the entire framework. We started by acknowledging that classification is complex, used Softmax to manage multiple outputs, and now we build the full pipeline using PyTorch's specialized tools.

You have learned that modern deep learning relies on a standard architecture pattern:

**Multi-Dimensional Input $\rightarrow$ Linear Layers $\rightarrow$ Non-linear Activations (Hidden) $\rightarrow$ Output Layer (Size $C$) $\rightarrow$ CrossEntropyLoss $\rightarrow$ Optimized Parameters.**