Project name "Raytrace interop"

Simple OpenGL/CUDA interop + simple raytracer.

This is “as is” and so on.

Demo of the result uploaded @ http://youtu.be/rb8kWRwIm1Y (beware of the youtube compression).

.cu/.cuh files are the CUDA files
.hpp files are the CPP files
.h files are the common for CUDA/CPP files.
*NB* The final result is not absolutely correct. Use at own risks.

RaytraceInterop has no copyright, no copyleft, use, sell and do whatever you want.

Naive implementation without shadow rays, brdfs, compex shading, etc.
Up to 40-50fps on Macbook Pro, i7, 650m, 512x512 resolution, can be at least 2-3 times faster.
