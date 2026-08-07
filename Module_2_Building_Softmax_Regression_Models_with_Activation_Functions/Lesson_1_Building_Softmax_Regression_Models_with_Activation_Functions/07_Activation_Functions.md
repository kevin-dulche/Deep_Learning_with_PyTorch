# Module 2: Activation Functions—The Non-Linear Engine

Welcome to a core module of deep learning! If linear layers ($\mathbf{w}\cdot\mathbf{x} + b$) are responsible for *scoring* inputs, **Activation Functions** are responsible for introducing the necessary non-linearity that allows the network to learn complex, curved patterns. Without them, no matter how deep your network is, it can only solve problems as if they were simple linear separations—which is insufficient for real-world data like images or speech.

The choice of activation function is perhaps one of the most impactful decisions you will make in model design, directly affecting training speed and even whether the model learns at all.

## The Problem: Gradient Flow
During backpropagation, the gradient signal (the update direction) must flow backward through every single layer. This process involves multiplying many local derivatives together. If these individual derivatives are consistently small or zero, two major problems occur:

1.  **Vanishing Gradients:** The product of tiny numbers quickly approaches zero, making updates negligible. The network effectively stops learning because the signal from early layers can't reach the initial weights ($\mathbf{w}$ and $b$).
2.  **Zero Gradients:** If a neuron outputs exactly zero for negative inputs (e.g., in ReLU), its gradient is zero, and it stops updating permanently—a condition known as **Dying Neurons**.

## Comparison of Key Activation Functions

Here we analyze the three most common functions used today:

### 1. Sigmoid ($\sigma$)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
*   **Output Range:** $(0, 1)$
*   **Pros:** Ideal for modeling probabilities in binary classification (as seen with logistic regression).
*   **Cons: The Vanishing Gradient Problem.** For very large positive or negative inputs ($|z|$), the derivative of $\sigma(z)$ rapidly approaches zero. Because gradients are multiplied layer by layer, this causes gradient signal dissipation, halting effective learning in deep models.

### 2. Hyperbolic Tangent ($\text{Tanh}$)
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
*   **Output Range:** $(-1, 1)$
*   **Advantage over Sigmoid:** $\text{Tanh}$ is **zero-centered**. Unlike Sigmoid (which always outputs positive values), Tanh’s output balances around zero. This characteristic helps Gradient Descent converge much faster and more stably than Sigmoid.
*   **Cons:** While better, it still suffers from the vanishing gradient problem for extreme inputs, though its maximum derivative is 1 (better than $0.25$ for Sigmoid).

### 3. Rectified Linear Unit ($\text{ReLU}$)
$$\text{ReLU}(z) = \max(0, z)$$
*   **Output Range:** $[0, \infty)$
*   **The Breakthrough Feature:** For any positive input ($z > 0$), the derivative is exactly **1**. This is revolutionary because it ensures that the gradient signal passes through unimpeded, significantly speeding up training in deep networks.
*   **Cons: The Dying ReLU Problem.** For negative inputs ($z \le 0$), $\text{ReLU}(z)$ outputs zero and its derivative is also zero. If a neuron consistently receives negative inputs (perhaps due to poor learning rates or data), it will output zero forever, ceasing all gradient flow—it "dies."

### Advanced Alternative: Swish
$$\text{Swish}(z) = z \cdot \sigma(z)$$
*   **Goal:** To fix the Dying ReLU problem.
*   **How it Works:** $\text{Swish}$ multiplies the input by its sigmoid activation. This design allows for small, non-zero gradients even when the input is negative, preventing neurons from completely dying out and generally leading to improved stability in optimization compared to basic ReLU.

| Function | Formula | Output Range | Key Advantage | Primary Drawback |
| :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z)$ | $[0, 1]$ | Good for binary probability output. | Vanishing gradients; not zero-centered. |
| **Tanh** | $\tanh(z)$ | $[-1, 1]$ | Zero-centered; faster convergence than Sigmoid. | Still prone to vanishing gradient. |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Derivative of 1 for $z>0$; solves vanishing gradients. | Dying ReLU problem (zero gradient for $z \le 0$). |
| **Swish** | $z \cdot \sigma(z)$ | Smoothly shaped | Addresses Dying ReLU; smoother gradients than ReLU. | Slightly more complex to compute. |

---

## PyTorch Implementation
In practice, we implement these functions in two ways:

1.  **Functional Calls (For the Forward Pass):** Using `torch` methods directly on a tensor:
    ```python
    output = torch.sigmoid(z)
    output_relu = torch.relu(z) 
    ```
2.  **Module Layers (For Model Definition):** Defining them as layers within an `nn.Sequential` block, which is the standard practice for model building:
    ```python
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim), 
        nn.ReLU(), # Use nn.ReLU() here
        nn.Linear(hidden_dim, out_dim)
    )
    ```

### Conclusion: The Modern Standard
In deep learning today, **ReLU** remains the most widely used default activation function for hidden layers due to its efficiency and ability to pass gradients unchanged. However, depending on the specific architecture or dataset, $\text{Tanh}$ and newer functions like Swish might provide superior performance by mitigating their respective gradient issues.