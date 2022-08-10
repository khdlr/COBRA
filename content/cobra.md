+++
title = "COBRA"
draft = false
+++

{{< figure src="architecture.png" caption=" ">}}

Following the idea of explicit contour prediction,
we have developed a new method called "Charting Outlines by Recurrent Adaptation" (COBRA).
It works by combining the idea of Active Contour models with deep learning.
First, a 2D CNN backbone derives feature maps from the input imagery.
Then, a 1D CNN (Snake Head) iteratively deforms an initial contour until to match the true contour.
