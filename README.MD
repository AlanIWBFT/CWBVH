
Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs
=============
This is my personal implementation of NVIDIA's paper "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs" (http://research.nvidia.com/publication/2017-07_Efficient-Incoherent-Ray). The program does hemisphere sampling with first hit rays, i.e. gathering, which is a primitive for many global illumination algorithms.

It loads mesh from mesh.dat and a list of surfels, which are defined by position, normal and radius, from samplecache.dat. The format is very easy to understand, just take a look at rtCore.cu. It then performs a simple surfel sorting pass and uses hemisphere gathering to calculate skylight irradiance for each surfel. While it should be faster to use shadow rays and importance sampling for this purpose, it's out of the scope of what we are trying to demonstrate.

The kernel is launched repeatedly for 10 times and the output is written to output.hdr.

Build
--------
Requires VS2015 and CUDA SDK v9.1 (the CUDA SDK version can be easily downgraded). Also needs tbb and embree3. embree3 is used as a static library (#define EMBREE_STATIC_LIB) but I think you can change it easily.

Performance
--------
Unfortunately I'm unable to reproduce the performance reported in the paper. Not only is the kernel not faster than the baseline kernel (Aila et. al. 2012), but *it has only 65% of the performace of baseline*. Please raise an issue if you have advice for that.

The program also includes the baseline kernel for comparison. Both of them are located in rtTrace.cu. 

There are some implementation differences between this and the paper. I didn't use the *Optimal SAH* for CWBVH construction (embree's BVH8 builder instead, which is a recursively splitting algorithm) and *auction algorithm* for child node ordering (greedy selection instead, which is >99% of the perf of auction). However according to the paper they should contribute <10% performance in total, so they should probably not be the reason for *5x* slow down compared to the paper.

Update 01/10/2019
--------
Finally I found the performance issue is caused by a really stupid mistake in the BVH converter. While I'm not going to update the codebase, I think it should be very easy for the reader to fix it by him/herself. After the fix is made, along with some micro optimizations to the ray-box intersection part (using PRMT, VMIN and VMAX to implement the code at the end of https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU), the implementation in this repository should be on par with the performance reported by the paper. I also compared it to NVIDIA's OptiX Prime and get very close performance, so I feel OptiX should be using CWBVH as well.
