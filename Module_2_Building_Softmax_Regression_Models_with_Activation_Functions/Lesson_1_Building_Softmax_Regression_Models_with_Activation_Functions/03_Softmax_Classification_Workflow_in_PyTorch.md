# Module 2 Practical Implementation: End-to-End Workflow on MNIST

In our previous modules, we mastered the theory—understanding how Softmax generalizes classification from two classes to $C$ classes, and recognizing that Cross-Entropy Loss provides the necessary smooth gradients for optimization. Now, it is time to put all those theoretical pieces together in a complete, working machine learning pipeline using PyTorch and the MNIST dataset.

Understanding this full workflow is critical because every step—from data loading to model dimensions—depends on the correct implementation of the previous one.

## Part I: The Data Pipeline (The Input)
Before training anything, we must load and prepare the data. We use the standard MNIST dataset, which consists of handwritten digits (0-9).

### 1. Loading and Transforming
We utilize `torchvision` to handle both loading and transformation seamlessly.

*   **Goal:** Load images from disk and convert their pixel values into PyTorch tensors.
*   **Mechanism:** We set up two datasets: one for training (`train=True`) and one for validation/testing (`train=False`). The crucial step is the `transforms.ToTensor()` function, which converts the raw image format into a floating-point tensor.
    *(Note: The resulting tensor shape for an MNIST image is typically $\text{[C, H, W]}$, i.e., $1 \times 28 \times 28$.)*

### 2. Preparing for Model Input (Flattening)
Our model's linear layer expects a single, flat feature vector. It cannot process the height and width dimensions separately yet.

*   **Action:** We must **flatten** the image tensor from $1 \times 28 \times 28$ down to a one-dimensional vector of $784$ features ($28 \times 28$).
*   **PyTorch Method:** When passing the input $\mathbf{x}$ through the model, we use `x.view(-1, 28 * 28)` to reshape the batch data correctly.

## Part II: Model Construction (The Architecture)
We build a custom class that defines our Softmax model architecture.

### 1. Defining the Class
```python
class SoftMax(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # The core component: one linear layer mapping features to classes
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 1. Compute raw scores (logits) using the learned weights W and biases b
        z = self.linear(x)  # z has shape [Batch_Size, Num_Classes]
        return z 
```

### 2. Dimensionality Check:
*   **Input Dimension (`input_dim`):** $28 \times 28 = 784$. (One score per pixel).
*   **Output Dimension (`output_dim`):** $10$ (One raw score for each of the 10 classes, 0 through 9).

The linear layer handles all the complexity: it calculates $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, where $\mathbf{W}$ is $10 \times 784$ and $\mathbf{b}$ is $10$.

### 3. Making a Prediction
Once we have the raw logit scores ($\mathbf{z}$) for an entire batch, predicting the class label requires only one function call:

```python
# Get the logits (raw scores)
z = model(x.view(-1, 28 * 28)) 

# Select the index of the largest score across the class dimension (dim=1)
predicted_class = torch.argmax(z, dim=1) 
```
The `torch.argmax(..., dim=1)` operation is what performs the final classification based on maximum score selection.

## Part III: The Training Loop Configuration

We set up our environment to train and evaluate the model efficiently.

### 1. Loss Criterion (Best Practice!)
Crucially, for multi-class problems, we use `nn.CrossEntropyLoss()`.
*   **Why this choice?** Unlike calculating Softmax manually and then using Negative Log Likelihood Loss (NLLLoss), PyTorch's `CrossEntropyLoss` is a built-in function that correctly combines the softmax operation *and* the cross-entropy loss into one stable, optimized call. This saves us from complex manual coding while ensuring numerical stability.
*   **Target Label Requirement:** Remember that `nn.CrossEntropyLoss()` expects target labels $\mathbf{y}$ to be of type `LongTensor` with shape $N$ (not $N \times 1$).

### 2. The Full Training Cycle
The standard PyTorch loop must execute six core steps for every batch:

1.  **Forward Pass:** Calculate logits $\mathbf{z}$.
2.  **Loss Calculation:** Compute the loss using `criterion(z, y)`.
3.  **Zero Gradients:** Clear previous gradients (`optimizer.zero_grad()`).
4.  **Backward Pass:** Calculate gradients for all parameters ($\frac{\partial \text{Loss}}{\partial w}, \dots$). $\rightarrow$ **`loss.backward()`**
5.  **Parameter Update:** Adjust weights and biases based on the calculated gradient direction and learning rate. $\rightarrow$ **`optimizer.step()`**

## Part IV: Prediction and Evaluation (The Result)

After training, we test the model's performance on unseen validation data. This involves a final loop that counts correct predictions:

1.  Pass the test input $\mathbf{x}_{\text{test}}$ through the model to get logits $\mathbf{z}$.
2.  Use `torch.argmax(z, dim=1)` to find the predicted class $\text{y}_{\text{hat}}$.
3.  Compare $\text{y}_{\text{hat}}$ element-wise with the true label $\mathbf{y}$ and calculate the accuracy (number of matches / total samples).

By following this structured workflow, you have successfully moved from raw images to trained, functional classification models capable of interpreting complex multi-class patterns.