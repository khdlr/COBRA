+++
title = "Results"
draft = false
+++

<div class="visual_row">
  <img src="./anim/11.svg"></img>
  <img src="./anim/16.svg"></img>
</div>

<div class="visual_row">
  <img src="./anim/21.svg"></img>
  <img src="./anim/25.svg"></img>
</div>

<div class="visual_row">
  <img src="./anim/28.svg"></img>
  <img src="./anim/30.svg"></img>
</div>

Existing approaches for calving front detection generally work by first performing
a pixel-wise segmentation or edge detection,
and then extract the actual calving front in a post-processing step.
Our goal in this study is to build a model that only needs a single step,
and directly outputs the calving front as a polyline.
