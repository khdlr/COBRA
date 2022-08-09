+++
title = "Idea"
draft = false
+++

{{< figure src="helheim_overlay.jpg" caption=" ">}}

Existing approaches for calving front detection generally work by first performing
a pixel-wise segmentation or edge detection,
and then extract the actual calving front in a post-processing step.
Our goal in this study is to build a model that only needs a single step,
and directly outputs the calving front as a polyline.
