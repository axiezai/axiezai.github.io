---
layout: post
title: 'Rubber Bands Tie Our Brains Together Too'
date: 2018-12-28 22:16:00
categories: jekyll update
mathjax: true
---
<p style = "font-family:Tahoma; text-align:center;">
Rubber bands in brains? The mysteries behind spectral clustering.
</p>

<p align = "center">
<img src = "https://imgs.xkcd.com/comics/rubber_sheet_2x.png">
<figcaption style = "font-size:65%">Source: xkcd.com/1158/</figcaption>
</p>

Hello, readers, look at your rubber band, now back to my rubber band, now back at your rubber band, now back to mine. Sadly, your rubber band isn't as cool as mine, but if your rubber band understood graph theory, it could be as cool as mine. Look down, back up, where are you? You're inside a brain with your rubber band. What's in your hand? Back at me, I have it, it's a spectral clustering algorithm. Look again, the brain's connected by my cool rubber bands. [I'm imaginging Old Spice commercials.](https://www.youtube.com/watch?v=owGykVbfgUE)

People use hair bands to bundle up their messy hair all the time. As it turns out, this simple act of using elasticity is similar to a clustering approach that groups together messy data. Here, I'd like to show some cool brains as an introduction to spectral clustering, and explain how a popular clustering algorithm can be simply viewed as perturbations of rubber bands.

My lab is often curious about how the brain's structure lead to function. In the case of human brains, the architecture of axons and their myelinated sheath facilitate the diffusion of water molecules along their main directions. By estimating the diffusion gradient in various ${x,y,z}$ directions, we can map out the underlying structure that connects specialized brain regions of interest. 


```python
from scipy.io import loadmat    # needed for loading .mat matlab files
import numpy as np              # for vector math
import os                       # for getting absolute path to files in your directory
import sklearn.cluster          # for spectral clustering (and other clustering methods)
import matplotlib.pyplot as plt # for plotting
from nilearn import plotting    # for plotting glass brains
import pandas as pd             # for easier data manipulation

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # ignore silly warnings
```