#include <stdlib.h>
#include <stdio.h>

/////////////////////////////////////////
//widh, height, indices of the params in the global misc array
#include "common_defines.h"

//precalculated halton random numbers
#include "halton.cuh"
//Scene, Sphere, Plane, Ray, Material
#include "structures.cuh"
//CUDA workarounds, 
#include "misc.cuh"
//intersections, matrices
#include "math.cuh"

#include "raytraceInterop.hpp"

typedef unsigned int uint;

__constant__ int MAX_REFL_DEPTH = 5;
//hard code light intensity
__constant__ float intense = 2;

__device__ int get_global_id(int d) {
    switch(d) {
        case 0:  return blockIdx.x * blockDim.x + threadIdx.x;
        case 1:  return blockIdx.y * blockDim.y + threadIdx.y;
        default: return 0;
    }
}

//***********************

//gives random float pair
__device__
float2 randomFloat2(uint2* state) {
	const uint a = (state->x++) % HALTON_NUMBERS;
	const uint b = (state->y++) % HALTON_NUMBERS;
    return make_float2(haltonX[a],haltonY[b]);
}

//this most likely is not correct, should give random point on a hemisphere
//the inversing of the direction is not working, should make transformation matrix according to the normal and rotate the points
__device__
void hemiSpherePoint(Scene* scene, float3* normal, float3* out, uint2* randomState, float* outPos) {
    float2 r = randomFloat2(randomState);
    
	float u = r.x;
	float v = r.y;
	
	float thetaSin = sqrt(u);
	float thetaCos = sqrt(1.0f - u);
	float phi = M_PI_F * 2.0f * v;
    
	*out = normalize(make_float3(cos(phi) * thetaCos, sin(phi) * thetaCos, thetaSin));
    
	if (dot(*out, *normal) < 0) {
		*out = *out * -1.f;
	}
    
	*out = normalize(*out);
}

//refract ray
__device__
bool refract(float3* V, float3* N, float3* out, float refrIndex) {
	float cosI = - dot(*N, *V);
	float sint2 = refrIndex * refrIndex * (1.f - cosI * cosI);
	if (sint2 > 1.f) return false;
	float cosT = sqrt(1.f - sint2);
	*out = refrIndex * *V + (refrIndex * cosI - cosT) * *N;
	return true;
}

//gives shlick approx
__device__
float shlick(float3* V, float3* N, float n1, float n2) {
	float r0 = (n1 - n2) / (n1 + n2);
	r0 *= r0;
	float cosX = -dot(*N, *V);
	if (n1 > n2) {
		float n = n1 / n2;
		float cosI = - dot(*N, *V);
		float sint2 = n * n * (1 - cosI*cosI);
		if (sint2 > 1.0f) return 1.0;
		cosX = sqrt(1.0 - sint2);
	}
	float x = 1.f - cosX;
	return r0 + (1.0 - r0) * x * x * x * x * x;
    
}

//reflects ray
__device__
void reflect(float3* ray, float3* normal, float3* out) {
    *out = *ray - 2.0f * dot(*ray, *normal) * (*normal);
}

__device__
void rayTrace(Ray* ray, Scene* scene, IntersectionData* intersectionData)  {
	intersectionData->inLight = false;
	intersect(scene, ray, intersectionData);
	if (intersectionData->inLight) {
		intersectionData->hit = false;
		intersectionData->inLight = true;
	}
}

//scene data hardcoded on the gpu
__device__
void initScene(Scene* scene, const float* miscData) {
	float reflection = 0.3f;
	float z = 150.f;
	int speed = 10;
    
	scene->spheresCount = 4;
	
	scene->spheres[0].origin.x =	 0.f	+ miscData[0] * speed;
	scene->spheres[0].origin.y =	 1000.f + miscData[1] * speed;
	scene->spheres[0].origin.z = z		+ miscData[2] * speed;
	scene->spheres[0].radius = 150.f;
	scene->spheres[0].material.diffuse.x = 1.f;
	scene->spheres[0].material.diffuse.y = 1.f;
	scene->spheres[0].material.diffuse.z = 1.f;
	scene->spheres[0].material.emission = 0;
    
	scene->spheres[0].material.reflection = 1.f;
    
	scene->spheres[1].origin.x = 150.f  +   0.f + miscData[3] * speed;
	scene->spheres[1].origin.y = 1100.f +   0.f + miscData[4] * speed;
	scene->spheres[1].origin.z =    z + miscData[5] * speed;
	scene->spheres[1].radius = 150.f;
	scene->spheres[1].material.diffuse.x = 1.f;
	scene->spheres[1].material.diffuse.y = 1.f;
	scene->spheres[1].material.diffuse.z = 1.f;
	scene->spheres[1].material.reflection = reflection;
	scene->spheres[1].material.emission = 0;
    
	scene->spheres[2].origin.x = 450.f;
	scene->spheres[2].origin.y = 1100.f;
	scene->spheres[2].origin.z =    100.f;
	scene->spheres[2].radius = 150.f;
	scene->spheres[2].material.diffuse.x = .5f;
	scene->spheres[2].material.diffuse.y = .5f;
	scene->spheres[2].material.diffuse.z = .5f;
	scene->spheres[2].material.reflection = reflection;
	scene->spheres[2].material.emission = 0;
    
	scene->spheres[3].origin.x = 250.f;
	scene->spheres[3].origin.y = 1100.f;
	scene->spheres[3].origin.z =    400;
	scene->spheres[3].radius = 100.f;
	scene->spheres[3].material.diffuse.x = .6f;
	scene->spheres[3].material.diffuse.y = 1.f;
	scene->spheres[3].material.diffuse.z = .3f;
	scene->spheres[3].material.reflection = reflection;
	scene->spheres[3].material.emission = 0;
    
	scene->spheres[4].origin.x = 0.f;
	scene->spheres[4].origin.y = 1100.f;
	scene->spheres[4].origin.z =   300.f;
	scene->spheres[4].radius = 150.f;
	scene->spheres[4].material.diffuse.x = 1.f;
	scene->spheres[4].material.diffuse.y = 1.f;
	scene->spheres[4].material.diffuse.z = 1.f;
	scene->spheres[4].material.reflection = reflection;
	scene->spheres[4].material.emission = 0;
    
	scene->planesCount = 6;
    
	//bottom
	scene->planes[0].normal = make_float3(0.f,0.f,1.f);
	scene->planes[0].point = make_float3(0.f,0.f,-100.f);
	scene->planes[0].material.diffuse.x = 1.0f;
	scene->planes[0].material.diffuse.y = 1.0f;
	scene->planes[0].material.diffuse.z = 1.0f;
	scene->planes[0].material.reflection = 0.f;
    
	//far
	scene->planes[1].normal = normalize(make_float3(0.f,-1.f,0.f));
	scene->planes[1].point = make_float3(0.f,1500.f,0.f);
	scene->planes[1].material.diffuse.x = 1.f;
	scene->planes[1].material.diffuse.y = 1.f;
	scene->planes[1].material.diffuse.z = 1.f;
	scene->planes[1].material.reflection = 0.0f;
    
	//left
	scene->planes[2].normal = normalize(make_float3(1.f,0.f,0.0f));
	scene->planes[2].point = make_float3(-700.f,0.f,0.f);
	scene->planes[2].material.diffuse.x =  1.f;
	scene->planes[2].material.diffuse.y =  0.f;
	scene->planes[2].material.diffuse.z =  0.f;
	scene->planes[2].material.reflection = 0.0f;
    
	//right
	scene->planes[3].normal = normalize(make_float3(-1.f,0.f,0.0f));
	scene->planes[3].point = make_float3(700.f,0.f,0.f);
	scene->planes[3].material.diffuse.x = 0.f;
	scene->planes[3].material.diffuse.y =  0.f;
	scene->planes[3].material.diffuse.z =  1.f;
	scene->planes[3].material.reflection = 0.0f;
    
	//top
	scene->planes[4].normal = normalize(make_float3(0.f,0.f,-1.0f));
	scene->planes[4].point = make_float3(0.f,0.f,1600.f);
	scene->planes[4].material.diffuse.x = 1.f;
	scene->planes[4].material.diffuse.y = 1.f;
	scene->planes[4].material.diffuse.z = 1.f;
	scene->planes[4].material.reflection = 0.0f;
    
	//back
	scene->planes[5].normal = normalize(make_float3(0.f,1.f,.0f));
	scene->planes[5].point = make_float3(0.f,-1500.f,0.f);
	scene->planes[5].material.diffuse.x = 1.f;
	scene->planes[5].material.diffuse.y = 1.f;
	scene->planes[5].material.diffuse.z = 1.f;
	scene->planes[5].material.reflection = 0.0f;
    
	scene->lightsCount = 1;
	scene->lights[0].intensity = 0.1f;
	scene->lights[0].origin = make_float3(0.f, 0.f, 1000.f);
	scene->lights[0].radius = 900;
}

extern "C"
__global__
void raytrace(cudaSurfaceObject_t image, const float* miscData, float* deviceRes)
{
    //consider move this in global mem
    
    //init scene data
    __shared__ Scene scene;
    __shared__ float16 matX;
	__shared__ float16 matZ;

    if (threadIdx.x==0) {
        initScene(&scene, miscData);
        
        //check for camera movement
        rotationX(miscData[PARAM_MOUSE_UP_DOWN]/100.f,    &matX);
        rotationZ(miscData[PARAM_MOUSE_LEFT_RIGHT]/100.f, &matZ);
    }
    
    __syncthreads();
    
    //get current pixel x and y
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    float3 resultColor = make_float3(0.f, 0.f, 0.f);
    
    //get the index in the final buffer
    const int index = (x + y * HEIGHT) * 4;
    
    const float iteration = (float)(miscData[PARAM_ITER])    ;
    
    //get some starting index for the halton sequence, not sure if this keeps it low discrapancy.
    //but it kinda works.
    int xored = x ^ y;
	uint2 randomState = make_uint2(317.f * (index/4) * miscData[PARAM_RAND] + iteration * (37.f + xored),
                                   227.f * (index/4) * miscData[PARAM_RAND2] + iteration * (51.f + xored));
    
    const int samples = miscData[PARAM_DEPTH];

    //generate ray
	Ray ray;
	ray.flags = RF_DIRECT;
	ray.origin.x =   0.f;
	ray.origin.y = -50.f;
	ray.origin.z =   0.f;
	ray.origin = ray.origin ;//+ make_float3(miscData[6], miscData[8] * 5, miscData[7]);
	
    const float subPixelWeight = 1.f / (float)(samples * samples);
    
	bool inLight = false;
    //do samples^2 samples
	for (int i = 0; i < samples; ++i) {
		for (int j = 0; j < samples; ++j) {
			IntersectionData intersectionData;
			intersectionData.hit = false;
            
            float2 r = randomFloat2(&randomState);
			float dx = float(x) + r.x;
			float dy = float(y) + r.y;
            
			ray.direction.x = (dx - WIDTH / 2 - ray.origin.x) / 10;
			ray.direction.y = -ray.origin.y;
			ray.direction.z = -(dy - HEIGHT / 2 - ray.origin.z) / 10;
			ray.direction = ray.direction;// + make_float3(miscData[6], miscData[8] * 5, miscData[7]);
            
			ray.direction = matrixVectorMultiply(&ray.direction, &matX);
			ray.direction = matrixVectorMultiply(&ray.direction, &matZ);
			ray.direction = normalize(ray.direction);
			
			rayTrace(&ray, &scene, &intersectionData);
            
			float3 tempColor = make_float3(1, 1, 1);
			
            //hard coded refr ior
            const float n1 = 1.5f;
			const float n2 = 1.4f;
            
            //hard coded max ray depth
            const int pathDepth = 7;

			for (int i = 0; i < pathDepth; ++i) {
				if (intersectionData.inLight) {
					inLight = true;
                    tempColor = tempColor * intense;
					break;
				}
                
                //no environment for now
				if (!intersectionData.hit) break;
				
				tempColor = tempColor * intersectionData.material->diffuse;
                
                //hard coded mtls
				if(intersectionData.type == GT_SPHERE && intersectionData.index == 0) {
					float f = shlick(&ray.direction, &intersectionData.normal, n1, n2);
                    float2 r = randomFloat2(&randomState);

                    if (r.x < f) {
						reflect(&ray.direction, &intersectionData.normal, &ray.direction);
					} else {
						refract(&ray.direction, &intersectionData.normal, &ray.direction, n1 / n2);
					}
				} else if (intersectionData.type == GT_SPHERE && intersectionData.index == 2)  {
					reflect(&ray.direction, &intersectionData.normal, &ray.direction);
				} else {
					float posibility;
					hemiSpherePoint(&scene, &intersectionData.normal, &ray.direction, &randomState, &posibility);
				}
                
				ray.origin = intersectionData.intersectionPoint + 0.0001f * ray.direction;
				rayTrace(&ray, &scene, &intersectionData);
			}
			if (inLight) {
				resultColor = (tempColor) * subPixelWeight * 0.75f;
			}
            
		}
	}

	float4 color = make_float4(resultColor.x, resultColor.y, resultColor.z, 1.f);
    
    //override or add to the result
    if (iteration == 1) {
        deviceRes[index + 0] = color.x;
        deviceRes[index + 1] = color.y;
        deviceRes[index + 2] = color.z;
        deviceRes[index + 3] = color.w;
    } else {
        deviceRes[index + 0] += color.x;
        deviceRes[index + 1] += color.y;
        deviceRes[index + 2] += color.z;
        deviceRes[index + 3] += color.w;
        
    }
    
    color.x = deviceRes[index + 0] / iteration;
    color.y = deviceRes[index + 1] / iteration;
    color.z = deviceRes[index + 2] / iteration;
    color.w = deviceRes[index + 3] / iteration;
    
    surf2Dwrite(color, image, x * sizeof(color), y, cudaBoundaryModeClamp);
}

