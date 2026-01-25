# K-Means Clustering

K-Means is a **distance-based unsupervised learning algorithm** used for
**clustering and pattern discovery** in unlabeled data.

It aims to partition data into **K distinct clusters** such that the
**within-cluster variance is minimized** and the **between-cluster
variance is maximized**.

K-Means is one of the most widely used algorithms in: 
- Customer segmentation\
- Market basket analysis\
- Image compression\
- Anomaly detection\
- Feature engineering pipelines

------------------------------------------------------------------------

## Core Intuition

K-Means answers the question:

> *"How can I group similar data points together without knowing the
> labels?"*

It does this by: - Assigning each point to the **nearest centroid** -
Updating centroids as the **mean of assigned points** - Repeating until
**convergence**

------------------------------------------------------------------------

## Key Concepts & Keywords

### Clusters

A **cluster** is a group of data points that are: - More similar to each
other - Less similar to points in other clusters

Formally, K-Means minimizes:

\[ `\sum`{=tex}*{i=1}\^{K} `\sum`{=tex}*{x `\in `{=tex}C_i} \|\|x -
`\mu`{=tex}\_i\|\|\^2 \]

### Centroids

A **centroid** is the **mean vector** of all points in a cluster:

\[ `\mu`{=tex}*i = `\frac{1}{|C_i|}`{=tex} `\sum`{=tex}*{x
`\in `{=tex}C_i} x \]

------------------------------------------------------------------------

## Standard K-Means Workflow

1.  Initialize centroids randomly\
2.  Assignment step: assign points to nearest centroid\
3.  Update step: recompute centroids\
4.  Repeat until convergence

This is an **Expectation-Maximization style loop**.

------------------------------------------------------------------------

## K-Means++

Improves initialization by selecting well-separated centroids.

Benefits: - Faster convergence\
- Better cluster quality\
- Reduced local minima

------------------------------------------------------------------------

## Choosing K

### Elbow Method

Plot K vs inertia and find elbow.

### Silhouette Score

Measures separation quality.

------------------------------------------------------------------------

## Use Cases

### Customer Segmentation

*Example:* (Insert Example Repo when completed)

### Image Compression

*Example:* (Insert Example Repo when completed)

### Anomaly Detection

*Example:* (Insert Example Repo when completed)

------------------------------------------------------------------------

## Pros

-   Simple
-   Fast
-   Scalable
-   Interpretable

## Cons

-   Must choose K
-   Sensitive to scale
-   Poor with non-globular clusters

------------------------------------------------------------------------

## Mental Model

> **Gradient descent on cluster centers**

------------------------------------------------------------------------

## Production Usage

Used for: - Feature engineering - Segmentation engines - Outlier
detection

------------------------------------------------------------------------
