# Module 4: Deep Neural Networks—Scaling Intelligence

Welcome! If previous modules taught us how to build single-layer models and understand the fundamentals of classification, this module takes us into **Deep Learning**. A deep neural network is not simply one with a lot of neurons; it refers specifically to an architecture that utilizes **multiple hidden layers**, allowing the model to process data through increasingly abstract feature representations.

## Part I: The Principle of Depth vs. Width
### 1. What Makes it "Deep"?
A network becomes *deep* when its input $\mathbf{x}$ must pass through two or more successive transformations (hidden layers) before reaching the output.

| Feature | Shallow Network | Deep Network | Advantage |
| :--- | :--- | :--- | :--- |
| **Hidden Layers** | One hidden layer ($H$). | Multiple hidden layers ($\mathbf{H}_1, \mathbf{H}_2, \dots$). | Improved model capacity and ability to learn hierarchical features. |
| **Complexity** | Limited by the single linear transformation. | Each layer learns increasingly abstract concepts from the previous output. | Solves highly complex, real-world pattern recognition tasks. |

### 2. Why Depth Improves Performance (The Hierarchy of Features)
Deep networks are powerful because they enable hierarchical feature extraction:

*   **Layer 1:** Learns simple, local features (e.g., edges, lines in an image).
*   **Layer 2:** Combines the local features from Layer 1 into more complex patterns (e.g., corners, textures, specific shapes).
*   **Deep Layers:** Combine these intermediate patterns into high-level concepts necessary for classification (e.g., a complete eye, a car wheel, or a full digit shape).

This sequential refinement allows the network to decompose a massive problem into smaller, manageable sub-problems—the essence of human intelligence.

## Part II: Deep Network Architecture in PyTorch
### 1. Defining Sequential Layers
When building a deep network in code, we simply chain together multiple linear layers and activation functions. The critical point is that **the output size of layer $L$ must match the input size of layer $L+1$.**

*   **Input $\mathbf{x}$:** Has dimension $d_{\text{in}}$.
*   **Hidden Layer 1 (H1):** Takes $d_{\text{in}}$ and outputs $h_1$.
*   **Hidden Layer 2 (H2):** Takes $h_1$ as its input and outputs $h_2$.
*   **Output Layer:** Takes $h_2$ as its input and outputs $C$ (number of classes).

**The PyTorch Implementation Logic:**
```python
class DeepNet(nn.Module):
    # ... __init__ method definition ...
    def forward(self, x):
        # 1. First Hidden Layer: Input -> h1 neurons
        x = self.linear1(x)
        x = torch.relu(x) # Activation 1 (e.g., ReLU)

        # 2. Second Hidden Layer: h1 -> h2 neurons
        x = self.linear2(x)
        x = torch.tanh(x) # Using a different activation function

        # 3. Output Layer: h2 -> C classes
        z_output = self.linear_out(x) 
        return z_output  # Return raw logits (for CrossEntropyLoss)
```

### 2. Flexibility of Activation Functions
The choice of activation function at each hidden layer is a hyperparameter that determines the *character* of the learned features:
*   **ReLU:** Excellent for maintaining strong gradient signals across deep layers.
*   **Tanh/Sigmoid:** Useful, but can slow training due to vanishing gradients.

## Part III: Training and Evaluation
The training workflow itself remains consistent regardless of depth—the power comes from the architecture, not the loop logic.

1.  **Loss Function:** Always use `nn.CrossEntropyLoss()`. This single function manages the Softmax conversion internally while minimizing the negative log-likelihood loss across all $C$ classes.
2.  **Optimizer:** Use an optimizer like `torch.optim.SGD` or `Adam` to update all the parameters ($\mathbf{W}$ and $\mathbf{b}$) across every single layer simultaneously.

By successfully training a deep network, we prove that the model can learn complex mappings—that it has effectively learned the multi-layered rules required to translate raw pixel data into abstract, human-interpretable features of classification.