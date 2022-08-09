+++
title = "COBRA"
draft = false
+++

{{< figure src="architecture.png" caption=" ">}}

Our approach for this task is called "Charting Outlines by Recurrent Adaptation" (COBRA).
It works by combining the idea of Active Contour models with deep learning.
First, a 2D CNN backbone derives feature maps from the input imagery.
Then, a 1D CNN (Snake Head) iteratively deforms an initial contour until to match the true contour.