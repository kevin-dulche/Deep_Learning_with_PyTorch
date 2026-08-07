### Setting the Stage: Why Deep Learning? (Module 1)

Before we dive into the code and build our first model, let’s take a moment to address a foundational question that has often tripped up beginners: **Why do we even need deep learning?**

While simple linear models can solve many classification tasks—and they are great for understanding the fundamentals—they quickly hit a wall when the relationship between input data and output is non-linear, or when the underlying problem structure is complex (like identifying objects in an image).

Deep learning, at its core, is just sophisticated function approximation. We are training a machine to model a highly complicated, unknown function $f(x)$ that maps our features ($x$) to the desired output ($\hat{y}$). The "deep" part simply refers to using multiple layers (non-linear transformations) to represent this mapping in an increasingly abstract and refined way.

But before we get into architectures, we must first talk about **measurement**. How do we know if the function our model has learned is good or bad? We need a loss function—a metric that quantifies the difference between what the model predicted ($\hat{y}$) and what the true answer was ($y$). This brings us to our very first critical concept:

**Mean Squared Error (MSE) vs. Classification.**

If you are predicting a continuous value—say, the price of a house—then MSE is an excellent choice. It calculates the average squared difference between your prediction and the actual value. The goal in optimization is simply to minimize this single number.

However, when we are doing classification, our output isn't a number; it's a *probability distribution* over several mutually exclusive categories (e.g., Dog, Cat, Car). These probabilities must sum up to 1.0. If we mistakenly treat these probabilities as continuous targets and use MSE, the optimization process breaks down, giving us misleading gradients that don't effectively guide the model toward the true probabilistic boundaries.

So, what does MLE do better? It correctly frames the problem as maximizing the likelihood of observing our given data under a specific probability model—which naturally leads us to the incredibly powerful and standard choice: **Cross-Entropy Loss**.

Over the next few hours, we will see how Cross-Entropy loss doesn't just minimize error; it teaches the model *how* probabilities should behave when solving multi-class problems. This fundamental shift in perspective is what unlocks the power of modern AI...