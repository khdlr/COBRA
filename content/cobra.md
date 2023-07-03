+++
title = "COBRA"
draft = false
+++

{{< figure src="helheim_overlay.jpg" caption=" ">}}

Existing approaches for calving front detection generally work by first performing
a pixel-wise segmentation or edge detection,
and then extract the actual calving front in a post-processing step.
Our goal in this study is to build a model that only needs a single step,
and directly outputs the calving front as a polyline.

{{< figure src="architecture.png" caption=" ">}}

Following the idea of explicit contour prediction,
we have developed a new method called "Charting Outlines by Recurrent Adaptation" (COBRA).
It works by combining the idea of Active Contour models with deep learning.
First, a 2D CNN backbone derives feature maps from the input imagery.
Then, a 1D CNN (Snake Head) iteratively deforms an initial contour until to match the true contour.
