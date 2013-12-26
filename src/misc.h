#ifndef __MISC_H__
#define __MISC_H__

namespace RaytraceInterop {
    
#define check(ans) { RaytraceInterop::_check((ans), __FILE__, __LINE__); }
    
inline double diffclock(clock_t clock1,clock_t clock2){
	double diffticks = clock1 - clock2;
	return (diffticks) / CLOCKS_PER_SEC;
}

inline void _check(cudaError_t code, char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

}//namespace RaytraceInterop

#endif