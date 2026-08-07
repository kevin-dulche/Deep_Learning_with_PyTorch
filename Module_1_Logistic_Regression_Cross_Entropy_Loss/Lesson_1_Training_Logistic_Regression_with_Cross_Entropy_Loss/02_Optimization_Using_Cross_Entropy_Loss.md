# Optimization Using Cross-Entropy Loss

# Module 1 Deep Dive: Optimization and Implementation

Welcome back! In our last session, we established *why* Cross-Entropy Loss is superior for classification by linking it to Maximum Likelihood Estimation. Today, we focus on the mechanics: how do these theoretical differences manifest in the optimization process, and how do we implement this entire cycle using PyTorch?

## Part I: The Mathematics of Gradient Descent

The core concept that must be understood here is that **the loss function determines if gradient descent can find good parameters.** If a loss function cannot provide useful gradients (a non-zero signal), the model cannot learn, regardless of how perfect the architecture is.

We examine three cost functions to illustrate this crucial point:

### 1. The Threshold Cost Function
*   **Calculation:** This uses a hard threshold ($\text{THR}$), which outputs a definitive 0 or 1 class prediction.
*   **Problem:** Because it relies on discrete class predictions, the cost changes *only* when an input crosses the decision boundary (e.g., from misclassified to correctly classified).
*   **Gradient Failure:** This creates **discontinuities and flat regions** in the loss surface. If a slight change in parameters does not change the number of errors (remaining in a flat region), the gradient becomes zero, causing the optimization process to stall—the optimizer has no update signal.

### 2. The Sigmoid Cost Function
*   **Calculation:** This uses the smooth sigmoid output ($\hat{y}$).
*   **Improvement:** Since $\sigma(z)$ is a continuous curve, the cost value changes smoothly across most of the parameter space. The gradient remains non-zero even when the prediction is wrong, allowing the optimizer to continue making meaningful adjustments.

### 3. Cross-Entropy Loss ($\mathcal{L}$)
*   **Calculation:** Derived from MLE, this loss function combines the smoothness of the sigmoid with the probabilistic rigor required for classification.
    $$\mathcal{L}(\mathbf{\theta})=-\frac{1}{N} \sum_{n=1}^{N} [y_n \log(\hat{y}_n) + (1 - y_n) \log(1 - \hat{y}_n)]$$
*   **Superiority:** The resulting contour plot for $\mathbf{w}$ and $b$ shows smooth, well-shaped contours across the entire surface. The loss only flattens out at the absolute minimum, ensuring a useful, non-zero gradient almost everywhere else in the parameter space.

---

## Part II: Implementing Logistic Regression in PyTorch

With theory established, we move to the practical implementation pipeline required for any machine learning task using PyTorch.

### A. Model Definition (The Architecture)
You have two standard ways to define your model structure:

**1. Using `nn.Sequential` (Simple Models):**
This method chains operations together and is ideal for straightforward, linear models like basic logistic regression.
```python
# Combines a Linear Layer (z=wx+b) followed by the Sigmoid activation
model = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

# Making a prediction: pass input x through the model
yhat = model(x)
```

**2. Using `nn.Module` (Custom Models):**
For more complex or multi-stage models, defining a custom class is preferred.
```python
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1) # Defines the weight and bias layers

    def forward(self, x):
        # Step 1: Calculate z=wx+b
        z = self.linear(x)
        # Step 2: Apply sigmoid to get probability yhat
        return torch.sigmoid(z)
```

### B. The Loss Criterion (The Metric)
While you *can* write the Cross-Entropy formula manually using PyTorch tensor operations, it is highly recommended to use the built-in functionality provided by PyTorch, as it handles numerical stability and edge cases for you.

**1. Manual Calculation (For Understanding):**
```python
# Equivalent to the mathematical cross-entropy formula:
out = -1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
```

**2. Best Practice (Using Built-in Class):**
PyTorch provides `nn.BCELoss()` (Binary Cross Entropy Loss), which computes this exact formula internally and efficiently:
```python
criterion = nn.BCELoss() 
# Note: BCELoss handles the necessary calculations automatically.
```

### C. The Optimizer (The Engine)
We need an optimization algorithm to adjust parameters based on the computed loss. We typically use **Stochastic Gradient Descent (SGD)**:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
# lr defines the step size taken in the direction of the negative gradient.
```

---

## Part III: The End-to-End Training Loop

The training process is a structured cycle that must be executed precisely within PyTorch. This loop repeats many times (epochs) to allow the model parameters ($\mathbf{w}$ and $b$) to iteratively approach the minimum of the loss surface.

1.  **Data Loading:** Use `DataLoader` to manage the data, typically processing it in small batches (`batch_size`).
2.  **Prediction:** Calculate $\hat{\mathbf{y}} = \text{model}(\mathbf{x})$.
3.  **Loss Calculation:** Determine how wrong the predictions are: $\text{loss} = \text{criterion}(\hat{\mathbf{y}}, \mathbf{y})$.
4.  **Clear Gradients (`optimizer.zero_grad()`):** Crucial step! Before calculating gradients for the current batch, we must zero out any gradient information accumulated from previous batches.
5.  **Backward Pass (`loss.backward()`):** This is where PyTorch magic happens. The function automatically calculates the partial derivatives ($\frac{\partial \text{Loss}}{\partial w}$ and $\frac{\partial \text{Loss}}{\partial b}$) for every single parameter in the model (the gradient).
6.  **Parameter Update (`optimizer.step()`):** The optimizer uses the calculated gradients to adjust the parameters according to the update rule: $w_{\text{new}} = w_{\text{old}} - (\text{learning\_rate} \times \text{gradient})$.

### Summary of the Cycle
The combination of `loss.backward()` and `optimizer.step()` is the engine that drives learning, allowing parameters to move down the smooth, well-defined Cross-Entropy loss surface until an optimal solution is found.