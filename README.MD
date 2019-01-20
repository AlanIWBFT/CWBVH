
Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs
=============
This is my personal implementation of NVIDIA's paper "Efficient Incoherent Ray Traversal on GPUs Through Compressed Wide BVHs" (http://research.nvidia.com/publication/2017-07_Efficient-Incoherent-Ray). The program does hemisphere sampling with first hit rays, i.e. gathering, which is a primitive for many global illumination algorithms.

It loads mesh from mesh.dat and a list of surfels, which are defined by position, normal and radius, from samplecache.dat. It then generates AO rays for each surfel. For showcasing, the rays are closest-hit rays rather than shadow rays.

Build
--------
Requires CUDA SDK v10.0 (the CUDA SDK version can be easily downgraded), tbb and embree3.

When compiling with VS2017, install v140 (VS2015) toolset for the CUDA SDK to function properly.

Optionally there is an OptiX Prime path for performance comparison - if you want to use it you need to specify OptiX include and library paths. I didn't find a good FindOptiX cmake package :(

Currently only Windows is tested, as NVCC on Linux seems not to like CUDA in .cpp files.

Performance
--------
After I fixed the stupid mistake in the BVH converter, the implementation in this repository should be on par with the performance reported by the paper. It also yields similar (slightly slower) level of performance compared to OptiX Prime, so I feel OptiX should be using CWBVH as well. Some micro optimizations to the ray-box intersection part (using PRMT, VMIN and VMAX to implement the code at the end of https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection) are not done, but it should be easy for the reader to implement them.

Performance on the included mesh.dat and samplecache.dat, with 32 rays per surfel, on my GTX1080:

60.946MS, 548.42MRays/s (rtTraceBVH2 No Dynamic Fetch)

37.714MS, 886.24MRays/s (rtTraceCWBVH Dynamic Fetch)

35.294MS, 947.00MRays/s (OptiXPrime)
