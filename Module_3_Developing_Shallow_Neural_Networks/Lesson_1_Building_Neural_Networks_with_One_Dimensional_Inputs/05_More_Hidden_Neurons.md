# Module 3: Hidden Neurons and Model Capacity

Welcome! In previous modules, we learned that complex data requires non-linear boundaries. This module tackles the ultimate question of model design: **How big should my network be?** We will explore the concept of **Model Capacity**—the representational power of a neural network—and how expanding hidden layers allows us to solve problems previously considered impossible for simple models.

## Part I: The Limit of Simple Models
### 1. Representational Power
Data rarely separates neatly along a single line or with just two curves. Real-world patterns are complex, box-shaped, or multi-segmented—they require high representational power to model accurately.

*   **The Constraint:** A network with limited capacity (e.g., only two hidden neurons) is mathematically constrained. It can only generate a decision function that is a weighted combination of just two sigmoid components ($\mathbf{w}_1\sigma(\mathbf{x}) + \mathbf{w}_2\sigma(\mathbf{x})$).
*   **The Failure:** If the true optimal boundary is complex (like a box shape), and the model only has enough "building blocks" for two lines, it cannot build that box. Attempts to shift or scale the limited function simply relocate the error—they don't resolve the underlying mathematical limitation.

### 2. The Solution: Expanding Capacity
The solution is to increase the number of hidden neurons.

**Each new neuron acts as an independent, specialized component.** Mathematically, when we add a third neuron ($N_3$), we are adding a weighted sigmoid function $\mathbf{w}_3 \cdot \sigma(\mathbf{x})$ to the final output calculation:

$$\text{New Output} = (\mathbf{w}_1\cdot\sigma(\mathbf{x})) + (\mathbf{w}_2\cdot\sigma(\mathbf{x})) + (\mathbf{w}_3\cdot\sigma(\mathbf{x}))$$

By stacking and aggregating these independent, non-linear components (each neuron contributing a unique *feature* to the decision boundary), the model's ability to approximate complex functions increases dramatically. It gains the **flexibility** to "mold" an intricate shape that perfectly matches the target data boundary.

---

## Part II: Implementation in PyTorch
When we increase the number of hidden neurons, we are fundamentally increasing the size of the weight matrix $\mathbf{W}_{\text{hidden}}$.

### 1. Using `nn.Module` (The Formal Way)
This method gives you explicit control over every step and is best for understanding data flow:

*   **Structure:** We define a class that holds multiple `nn.Linear` layers, one for each neuron group.
*   **Forward Pass:** The input $\mathbf{x}$ passes through the first linear layer, followed by activation. This output then feeds into the next linear layer, and so on.

### 2. Using `nn.Sequential` (The Streamlined Way)
For simple stacking of layers, this is cleaner and less code:
```python
# A model with 7 hidden neurons
model = nn.Sequential(
    nn.Linear(input_dim, 7), # Layer 1: Maps to 7 features
    nn.Sigmoid(),            # Activation 1
    nn.Linear(7, output_dim),  # Layer 2: Uses the 7 features for final scoring
    nn.Sigmoid()             # Final activation
)
```

The `nn.Sequential` module automatically stacks these operations into a single, continuous pipeline, making it extremely concise to build deep architectures.

## Part III: Training and Optimization (The Process)
Increasing the capacity requires adjusting our training loop and optimizers:

1.  **Initialization:** We initialize all weights ($\mathbf{W}$) randomly. The model starts with zero knowledge—its predictions are meaningless noise.
2.  **Training Loop:** The optimization process remains constant (Cross-Entropy Loss $\rightarrow$ `loss.backward()` $\rightarrow$ `optimizer.step()`). The optimizer now has a much larger parameter space to explore, meaning it must adjust thousands of weights instead of just hundreds.
3.  **The Learning Goal:** During training, the network is forced to learn *which* combination and arrangement of sigmoid components are necessary to perfectly fit the data's complex boundaries, gradually transforming random noise into meaningful, feature-specific patterns.

## Conclusion: Mastering Model Capacity
By mastering the concept of hidden neurons, you gain two crucial skills:

1.  **Theoretical Depth:** Understanding that model complexity is not just about adding more layers, but about giving the network enough unique components (neurons) to accurately approximate the true underlying function $f(\mathbf{x})$.
2.  **Practical Skill:** Knowing how to implement this increased capacity using both the custom `nn.Module` and the streamlined `nn.Sequential`, allowing you to build highly versatile models for any complex task.