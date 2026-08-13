# Module 3: Forward Propagation—Making the Network Work

This module connects all our theoretical knowledge into one cohesive, functional piece of code. Forward propagation is the core process by which data flows through a neural network, transforming raw input features ($\mathbf{x}$) into a final prediction ($\hat{\mathbf{y}}$). Understanding this flow is key to debugging and designing any model.

## Part I: The Concept of Forward Propagation
Forward propagation is simply the standardized sequence of operations that takes an input signal through the entire network, layer by layer, until it produces an output prediction.

The general structure at any point in a layer is always the same: **Linear Transformation $\rightarrow$ Activation Function**.

### 1. The Process Flow
A neural network operates as a sequential pipeline:

$$\mathbf{x} \xrightarrow{\text{Linear}_1} \mathbf{z}^{(1)} \xrightarrow{\text{Activation}_1} \mathbf{a}^{(1)} \xrightarrow{\text{Linear}_2} \mathbf{z}^{(2)} \xrightarrow{\text{Activation}_2} \hat{\mathbf{y}}$$

*   **Input $\mathbf{x}$:** The initial data features.
*   **$\mathbf{z}^{(\text{layer})}: $ Logits (Raw Scores):** The output of the linear transformation ($\mathbf{w}\cdot\mathbf{x} + b$). These scores can be any real number, positive or negative.
*   **$\mathbf{a}^{(\text{layer})}:$ Activation Output:** The result after passing $\mathbf{z}$ through an activation function (e.g., ReLU, Sigmoid). This is the "signal" passed to the next layer.

### 2. Matrix Operations and Batching
When dealing with multiple samples in a batch ($\mathbf{X}_{\text{batch}}$), all the operations must be executed efficiently using matrix algebra:

*   If $\mathbf{X}$ is an $N \times F$ matrix (N samples, F features):
    1.  **Linear Transform:** The output logits are calculated as $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$. This single operation computes all scores for all $N$ samples across all $C$ classes simultaneously.
    2.  **Activation/Prediction:** The activation function is applied element-wise to the resulting matrix $\mathbf{Z}$, generating $\hat{\mathbf{Y}}$.

---

## Part II: Implementing Layers in PyTorch (`nn.Module`)

To make a model reusable and manageable, we must encapsulate its structure and logic within a custom Python class inheriting from `nn.Module`.

### 1. The `__init__` Method (The Blueprint)
This method defines the layers (the weights $\mathbf{W}$ and biases $\mathbf{b}$). We tell PyTorch what size of input and output each layer expects:
```python
def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    # Layer 1: Maps from input features to hidden neurons (Linear)
    self.linear_hidden = nn.Linear(input_dim, hidden_dim)
    # Layer 2: Maps from hidden neurons to final class scores (Linear)
    self.linear_output = nn.Linear(hidden_dim, output_dim)
```

### 2. The `forward` Method (The Execution)
This method defines the data flow—the actual steps of propagation.
1.  **First Layer:** Apply linear transformation $\rightarrow$ pass through activation ($\mathbf{a}^{(1)}$).
2.  **Second Layer:** Use $\mathbf{a}^{(1)}$ as input to the next linear transformation $\rightarrow$ apply final activation ($\hat{\mathbf{y}}$).

```python
def forward(self, x):
    # 1. Linear transform: x -> z_hidden
    z_hidden = self.linear_hidden(x)
    # 2. Activation (Non-linearity): z_hidden -> a_hidden
    a_hidden = torch.sigmoid(z_hidden) 
    # 3. Output linear transform: a_hidden -> z_output
    z_output = self.linear_output(a_hidden)
    # 4. Final Activation (Prediction): z_output -> y_hat
    return torch.sigmoid(z_output) # For classification probability
```

### 3. Using `nn.Sequential` (The Shortcut)
For models with a simple, straightforward sequence of operations, `nn.Sequential` is an efficient shortcut that builds and executes the pipeline automatically:
```python
model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim), 
    nn.Sigmoid(),     # Activation applied after this layer
    nn.Linear(hidden_dim, out_dim), 
    nn.Sigmoid()      # Final activation
)
```

---

## Part III: Classification vs. Regression (The Output Layer Decision)

While the network structure remains largely the same, the final layers and loss function change depending on the problem type.

| Feature | Multi-Class Classification | Single-Value Regression |
| :--- | :--- | :--- |
| **Goal** | Determine which of $C$ classes is most likely (probability distribution). | Predict a single continuous number (e.g., temperature, price). |
| **Output Layer Size** | Must equal the number of classes ($C$). | Must be 1. |
| **Final Activation** | Typically Softmax (or Sigmoid for binary) to produce probabilities. | Often *no* final activation is needed; raw linear output $\mathbf{z}$ is passed directly. |
| **Loss Function** | `nn.CrossEntropyLoss()` | Mean Squared Error (`nn.MSELoss()`) |

The key principle: The structure (hidden layers, linearity) remains the same, but the choice of loss function and the final activation determines whether the model predicts a probability distribution or a continuous value.