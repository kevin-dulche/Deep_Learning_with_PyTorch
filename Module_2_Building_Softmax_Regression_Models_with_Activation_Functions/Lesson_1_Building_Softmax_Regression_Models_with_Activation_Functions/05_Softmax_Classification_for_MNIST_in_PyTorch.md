# Module 2 Practical Deep Dive: Training and Evaluation

This video synthesizes everything we have learned—the architecture, the loss function, the optimization cycle, and the prediction method—into one complete, executable pipeline. We will train a Softmax classifier on the MNIST dataset to demonstrate how a model learns meaningful patterns from random noise.

## Part I: The Training Workflow Review
The core principle remains the same: **structured data flow $\rightarrow$ optimized learning**.

### 1. Data Preparation Revisited (Input)
*   **Goal:** Load and batch the MNIST dataset.
*   **Key Step:** Image tensors are flattened from $28 \times 28$ to a single row of $784$ features using `x.view(-1, 28 * 28)`. This ensures the input tensor matches the expected format of our linear layer.

### 2. The Softmax Classifier
The model remains defined by one core component: a linear layer mapping the flattened inputs (784) to the number of classes (10).
$$\mathbf{z} = \text{Linear}( \text{Flatten}(\mathbf{x}))$$
*   The output $\mathbf{z}$ is a vector of 10 raw **logits**—one score for each digit class.

### 3. The Optimization Cycle in Depth (Training)
The training loop runs the optimization steps across every batch of data, gradually minimizing the Cross-Entropy Loss:

$$\text{Train Loop Steps}: \quad \mathbf{\text{optimizer.zero\_grad}()} \rightarrow \mathbf{z = model}(\mathbf{x}) \rightarrow \mathbf{\text{loss}} = \text{criterion}(z, y) \rightarrow \mathbf{\text{loss.backward}()} \rightarrow \mathbf{\text{optimizer.step}()}$$

By repeating this cycle over many epochs, the weights ($\mathbf{W}$) and biases ($\mathbf{b}$) are iteratively adjusted to make the raw logit score for the correct class much higher than the scores for all incorrect classes.

## Part II: Evaluation and Interpretation (Validation)

Training is only half the battle; evaluation determines if the model actually learned anything useful.

### 1. Predicting on Test Data
We use the same prediction method as before, but this time we evaluate against a held-out test set ($\mathbf{x}_{\text{test}}$).

*   **Step A: Logit Calculation:** $\mathbf{z} = \text{model}(\mathbf{x}_{\text{test}})$.
*   **Step B: Prediction:** Use `torch.max(z.data, 1)` to find the index of the highest score for every sample in the batch. This gives us the predicted class $\hat{\mathbf{y}}$.

### 2. Calculating Validation Accuracy
We compare our prediction vector ($\hat{\mathbf{y}}$) against the true label vector ($\mathbf{y}_{\text{test}}$).

*   **Comparison:** The boolean comparison $\hat{\mathbf{y}} == \mathbf{y}_{\text{test}}$ yields a tensor of `True` (match) or `False` (mismatch).
*   **Counting:** Using `.sum()` on this Boolean tensor counts the number of matching predictions.
*   **Accuracy:** We finalize the metric by dividing the total correct count by the total number of test samples ($N_{\text{test}}$): $\text{Accuracy} = \frac{\text{Correct Predictions}}{N_{\text{test}}}$.

### The Meaningful Outcome: Weight Visualization
The most profound moment in this workflow is visualizing the model's internal parameters after training.

*   **Start:** Before any training, the weights ($\mathbf{W}$) are initialized randomly—visually appearing as random noise when viewed pixel-by-pixel.
*   **After Training:** The trained weight matrix $\mathbf{W}$ (viewed class by class) no longer looks random! Each row now contains patterns that resemble the features of a specific digit class (e.g., weights for Class 0 look like loops, weights for Class 7 show vertical lines).

This visual transformation from **random noise to recognizable pattern** is definitive proof: The model has successfully learned meaningful features and representations required to distinguish between all ten digits.