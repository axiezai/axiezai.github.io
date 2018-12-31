---
layout: post
title: 'Rubber Bands Tie Our Brains Together Too'
date: 2018-12-28 22:16:00
categories: jekyll update
mathjax: true
---
<p style = "font-family:Tahoma; text-align:center;">
Rubber bands in brains? Brief intro to graph representations and the mysteries behind spectral clustering.
</p>

<p align = "center">
<img src = "https://imgs.xkcd.com/comics/rubber_sheet_2x.png">
<figcaption style = "font-size:65%">Source: xkcd.com/1158/</figcaption>
</p>

Hello, readers, look at your rubber band, now back to my rubber band, now back at your rubber band, now back to mine. Sadly, your rubber band isn't as cool as mine, but if your rubber band understood graph theory, it could be as cool as mine. Look down, back up, where are you? You're inside a brain with your rubber band. What's in your hand? Back at me, I have it, it's a spectral clustering algorithm. Look again, the brain's connected by my cool rubber bands. [I'm imaginging Old Spice commercials.](https://www.youtube.com/watch?v=owGykVbfgUE)

People use hair bands to bundle up their messy hair all the time. As it turns out, this simple act of using elasticity is similar to a clustering approach that groups together messy data. Here, I'd like to show some cool brains as an introduction to spectral clustering, and explain how a popular clustering algorithm can be simply viewed as perturbations of rubber bands.

My lab is often curious about how the brain's structure lead to function. In the case of human brains, the architecture of axons and their myelinated sheath facilitate the diffusion of water molecules along their main directions. By estimating the diffusion gradient in various ${x,y,z}$ directions, we can map out the underlying fiber structure that connects specialized brain regions of interest. The raw data is simply a bunch of estimated vectors in the ${x,y,z}$ directions:

$$
\begin{Bmatrix}
x_1 & x_2 & ... & x_n \\ 
y_1 & y_2 & ... & y_n \\
z_1 & z_2 & ... & z_n
\end{Bmatrix}
$$

We can get artsy with these vectors and draw out all the fiber connections inside the brain by following the diffusion parameters voxel by voxel, publishers love these colorful images:

<p align = "center">
<img src = "/figures/b03_dti.png" style = "width:200px">
<figcaption style = "font-size:65%; text-align:center">White matter fibers inside a human brain</figcaption>
</p>

Unfortunately, instead of the colorful image, researchers get a matrix of numbers representing a weighted scalar of the connection strength between pairs of brain regions:

<p align = "center">
<img src = "/figures/b03_fs2_scmatrix.png" style = "width:350px">
<figcaption style = "font-size:65%; text-align:center;"> Structural connectivity matrix, notice how the intra-hemisphere connections are dense (near the diagonal), whereas the inter-hemisphere connections are sparse (off-diagonal squares). Colors inside the matrix represent the connectivity strength between regions. </figcaption>
</p>

The obvious flaw in this matrix is that there is no spatial information about the data points. We only know the connection strengths between regions, but we do not know the location of each brain region, and we have no coordinate system defining the relative distance between each adjacent brain region. While this is a fatal weakness in our data, by utilizing rubber bands, i.e., spectral clustering, we can group these data points onto a graph representation of the brain without spatial information. 

## Basic graph representation and graph notation
Given this set of $N$ brain regions $x_1, ..., x_N$ and connection strength $c_{i,j} \geq 0$ between all pairs of brain regions $x_i$ and $x_j$, we can transform this data into a simple representation on a graph: $G = (V,E)$. Each node $V_i$ in this graph ($G$) represents a brain region $x_i$. Two nodes are are connected if the connection strength $c_{i,j}$ between the brain regions $x_i$ and $x_j$ is bigger than zero (or a threshold), then the edge (E) between the two nodes is weighted by $c_{i,j}$. 

<p align = "center">
<img src = "https://upload.wikimedia.org/wikipedia/commons/2/2f/Small_Network.png" style = "width:250">
<figcaption style = "font-size:65%; text-align : center;">Wikipedia has a nice image visualizing the nodes (vertices) and edges of a graph: https://en.wikipedia.org/wiki/Vertex_(graph_theory)</figcaption>
</p>

Additionally, for each node/vertex $V_i \in V$ (English: $V_i$ inside of V), we define the <em>degree</em> of that node as the sum of the weights to adjacent nodes: 

$$
\begin{align*}
d_i = \sum_{j=1}^{N} c_{i,j}
\end{align*}
$$

And the <em>degree matrix</em> $D$ is defined as an N-dimensional diagonal matrix where the degrees $d_i, ..., d_N$ is on the diagonal. This is important because we need to use the degree matrix $D$ to construct <em>graph Laplacians</em> $L$, these Laplacians will be the input to our spectral clustering algorithm. We want these Laplacians because they have "nice" properties like 1) $L$ is symmetric and positive semi-definite, 2) $0$ is an eigenvalue of $L$ and it's corresponding eigenvector is a constant vector of ones, or 3) $L$ has non-negative, real-valued eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_N$. These properties are shared by graph Laplacians regardless of how you define your $L$:

$$
\begin{gathered}
L = D - C \\
L = I - D^{-1/2} C D^{-1/2} \\
L = I - D^{-1} C
\end{gathered}
$$

After transforming our data into the graph Laplacian, we can rephrase the clustering problem: **we want to find a partition of the graph such that the edges between different groups have low weights (different clusters are less likely to be associated with each other), and the edges within a group have high weights (data points within the same cluster are closely associated).** To keep our sanities in tact, this notation is sufficient for our purpose, I am neglecting some details such as directionality ($c_{i,j} = c_{j,i} \geq 0$) and local neighborhood relationships inside the graph.

## Spectral clustering


## Rubber band interpretation

```python
spectral   = sklearn.cluster.SpectralClustering(n_clusters=2, affinity='precomputed')
patient_ID = 0
spectral.fit(kernelized_matrices[patient_ID])
color = spectral.labels_
plot_glass_brain(color)
```

### Generalized spectral clustering algorithm: