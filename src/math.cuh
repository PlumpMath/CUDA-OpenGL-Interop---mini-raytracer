#ifndef __MATH_CUH__
#define __MATH_CUH__

__constant__ float INVALID_VALUE = 2e32f;

//intersects ray with plane
__device__
bool rayPlane(const Plane* p, const Ray* ray, float* t) {
	float dotProduct = dot(ray->direction, p->normal);
    if (dotProduct == 0.f) {
        return false;
    }
    *t = dot(p->normal, (p->point - ray->origin)) / dotProduct ;
    
    return *t >= 0.f;
}

//intersects ray with sphere
__device__
bool raySphere(Sphere* s, Ray* ray, float* t) {
	float3 rayToCenter = s->origin - ray->origin ;
	float dotProduct = dot(ray->direction,rayToCenter);
	float d = dotProduct * dotProduct - dot(rayToCenter, rayToCenter) + s->radius * s->radius;
    
	if (d < 0.f)
		return false;
    
	*t = (dotProduct - sqrt(d) );
	
	if (*t < 0.f) {
		return false;
	}
	return true;
}

//intersects ray with sphere (which is also light)
__device__
bool rayLight(PointLight* s, Ray* ray, float* t) {
	float3 rayToCenter = s->origin - ray->origin ;
	float dotProduct = dot(ray->direction, rayToCenter);
	float d = dotProduct * dotProduct - dot(rayToCenter, rayToCenter) + s->radius * s->radius;
    
	if (d < 0.f)
		return false;
    
	*t = (dotProduct - sqrt(d) );
	
	if (*t < 0.f) {
		return false;
	}
	return true;
}

//makes rotation matrix
__device__
void rotationMatrix(float x, float y, float z, float16* matrix) {
	set(matrix, cos(y)*cos(z),  -cos(x)*sin(z)+sin(x)*sin(y)*cos(z),    sin(x)*sin(z)+cos(x)*sin(y)*cos(z), 0.f,
                        cos(y)*sin(z), cos(x)*cos(z) + sin(x)*sin(y)*sin(z), -sin(x)*cos(z) + cos(x)*sin(y)*sin(z), 0.f,
                        -sin(y),                        sin(x)*cos(y),                         cos(x)*cos(y), 0.f,
                        0.f,                                  0.f,                                   0.f, 0.f);
}

//makes X rotation matrix
__device__
void rotationX(float x, float16* matrix) {
	set(matrix, 1.f,      0.f,      0.f, 0.f,
                           0.f,    cos(x), -sin(x), 0.f,
                           0.f,    sin(x),  cos(x), 0.f,
                           0.f,      0.f,      0.f, 0.f);
}

//makes Y rotation matrix
__device__
void rotationY(float y, float16* matrix) {
	set(matrix, cos(y), 0.f, sin(y), 0.f,
                           0.f, 1.f,    0.f, 0.f,
                           -sin(y), 0.f, cos(y), 0.f,
                           0.f, 0.f,    0.f, 0.f);
}

//makes Z rotation matrix
__device__
void rotationZ(float z, float16* matrix) {
	set(matrix, cos(z), -sin(z), 0.f, 0.f,
                           sin(z),  cos(z), 0.f, 0.f,
                           0.f,       0.f, 1.f, 0.f,
                           0.f,       0.f, 0.f, 0.f);
}

//gives m*v
__device__
float3 matrixVectorMultiply(float3* vector, float16* matrix) {
    float3 result;
    result.x = matrix->data[0]*(vector->x)+matrix->data[4]*(vector->y)+matrix->data[8]*(vector->z);
    result.y = matrix->data[1]*(vector->x)+matrix->data[5]*(vector->y)+matrix->data[9]*(vector->z);
    result.z = matrix->data[2]*(vector->x)+matrix->data[6]*(vector->y)+matrix->data[10]*(vector->z);
	return result;
}

//*****************************

//intersects ray with scene
__device__
bool intersect(Scene* scene, Ray* ray, IntersectionData* out) {
	float minT = INVALID_VALUE;
	float t;
	
	for (int i = 0; i < scene->spheresCount; ++i) {
		if (raySphere(&(scene->spheres[i]), ray, &t) == true) {
			if (t < minT) {
				out->type = GT_SPHERE;
				out->index = i;
				minT = t;
			}
		}
	}
	
	for (int i = 0; i < scene->planesCount; ++i) {
		if (rayPlane(&(scene->planes[i]), ray, &t) == true) {
			if (t < minT) {
				out->type = GT_PLANE;
				out->index = i;
				minT = t;
			}
		}
	}
    
	for (int i = 0; i < scene->lightsCount; ++i) {
		if (rayLight(&scene->lights[i], ray, &t) == true) {
			if (t < minT) {
				out->type = GT_LIGHT;
				out->index = i;
				minT = t;
			}
		}
	}
	
	if (minT == INVALID_VALUE || out->type == GT_INVALID) {
		out->hit = false;
		return false;
	}
    
	out->intersectionPoint = ray->origin + ray->direction * minT;
    
	if (out->type == GT_SPHERE) {
		out->material = &(scene->spheres[out->index].material);
		out->normal = normalize(out->intersectionPoint - scene->spheres[out->index].origin);
	} else if (out->type == GT_PLANE) {
		out->material = &(scene->planes[out->index].material);
		out->normal = scene->planes[out->index].normal;
	} else if (out->type == GT_LIGHT) {
		out->inLight = true;
	}
	
	out->t = minT;
	out->hit = true;
	return true;
}

#endif //__MATH_CUH__