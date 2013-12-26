#ifndef __MISC_CUH__
#define __MISC_CUH__

//since when I ported this form OpenCL some of the code wasn't compiling,
//I implemented the operators I needed here as a workaround

__device__
float length(float3 n) {
    return sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
}

__device__
float3 normalize(float3 n) {
    const float len = length(n);
    const float3 result = make_float3(n.x/len, n.y/len, n.z/len);
    return result;
}

__device__
float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__
float4 operator+(float4 a, float4 b){
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__
float3 operator*(float3 a, float3 b){
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__
float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__
float3 operator*(float a, float3 b){
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__
float3 operator*(float3 a, float b){
    return b*a;
}

__device__
float3 operator/(float3 a, float b){
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__device__
float4 operator/(float4 a, float b){
    return make_float4(a.x/b, a.y/b, a.z/b, a.w/b);
}

__device__
float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;	
}

__device__
float3 cross(float3 a, float3 b){
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__
bool alike(float3* col, float val) {
	if (fabs(col->x - val) < 0.000001f &&
		fabs(col->y - val) < 0.000001f &&
		fabs(col->z - val) < 0.000001f) return true;
	return false;
}

//in case anyone needs to draw normals
__device__
float3 normal2Color(float3 norm) {
	norm = normalize(norm);
	norm = norm + make_float3(1.f, 1.f, 1.f);
	norm = norm / 2.f;
	return norm;
}

#endif //__MISC_CUH__