#ifndef __STRUCTURES_CUH__
#define __STRUCTURES_CUH__


struct Material {
	float3 diffuse;
	float emission;
	float alpha;
	float reflection;
	float refraction;
};

enum RayFlags {
	RF_DIRECT = 1 << 0,
	RF_REFLECTION = 1 << 1,
	RF_REFRACTION = 1 << 2,
	RF_SHADOW = 1 << 3,
	RF_GI = 1 << 4,
	RF_GLOSSY = 1 << 5,
	RF_INVALID = 1 << 5,
	RF_LAST = 1 << 7
};

enum GeometryType {
	GT_SPHERE = 1 << 0,
	GT_PLANE = 1 << 1,
	GT_INVALID = 1 << 2,
	GT_LIGHT = 1 << 3,
	GT_LAST = 1 << 4,
};

struct PointLight {
	float3 origin;
	float radius;
	float intensity;
};

struct Ray {
	float3 origin;
	float3 direction;
	int flags;
};

struct Sphere {
    Material material;
	float3 origin;
	float radius;
};

struct Plane {
    Material material;
	float3 normal;
	float3 point;
};

struct Scene {
	Sphere spheres[10];
	int spheresCount;
	Plane planes[10];
	int planesCount;
	PointLight lights[10];
	int lightsCount;
};

struct IntersectionData {
    Material* material;
	float3 normal;
	float3 intersectionPoint;
    float t;
    int index;
    enum GeometryType type;
	bool inLight;
    bool hit;
};

__device__
struct float16 {
    __device__ float16(){}
    __device__ float16(float a, float b, float c, float d,
            float e, float f, float g, float h,
            float i, float j, float k, float l,
            float m, float n, float o, float p) {
        data[0] = a; data[1] = b; data[2] = c; data[3] = d;
        data[4] = e; data[5] = f; data[6] = g; data[7] = h;
        data[8] = i; data[9] = j; data[10] = k; data[11] = l;
        data[12] = m; data[13] = n; data[14] = o; data[15] = p;
        
    }
   __device__  float16(const float16& rhs) {
        memcpy(data, rhs.data, 16 * sizeof(float));
    }
    float data[16];
};

__device__
void set(float16* matrix,float a, float b, float c, float d,
         float e, float f, float g, float h,
         float i, float j, float k, float l,
         float m, float n, float o, float p) {
    matrix->data[0] = a; matrix->data[1] = b; matrix->data[2] = c; matrix->data[3] = d;
    matrix->data[4] = e; matrix->data[5] = f; matrix->data[6] = g; matrix->data[7] = h;
    matrix->data[8] = i; matrix->data[9] = j; matrix->data[10] = k; matrix->data[11] = l;
    matrix->data[12] = m; matrix->data[13] = n; matrix->data[14] = o; matrix->data[15] = p;
}

#endif //__STRUCTURES_CUH__