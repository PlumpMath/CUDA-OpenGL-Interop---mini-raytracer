#ifndef __COMMON_DEFINES_H__
#define __COMMON_DEFINES_H__

//index of the param in the misc params array (both host and device)
#define PARAM_SPHERE1_LEFT_RIGHT 0
#define PARAM_SPHERE1_FRONT_BACK 1
#define PARAM_SPHERE1_UP_DOWN    2

#define PARAM_SPHERE2_LEFT_RIGHT 3
#define PARAM_SPHERE2_FRONT_BACK 4
#define PARAM_SPHERE2_UP_DOWN    5

#define PARAM_MOUSE_LEFT_RIGHT   9
#define PARAM_MOUSE_UP_DOWN     10

#define PARAM_WIDTH             11
#define PARAM_HEIGHT            12
#define PARAM_RAND              13
#define PARAM_ITER              14
#define PARAM_DEPTH             15
#define PARAM_RAND2             16
#define PARAM_LAST              17

//well, keep this power of 2
#define HEIGHT 512
#define WIDTH 512

#define M_PI_F 3.14159265358979323846f

#define HALTON_NUMBERS 4096

#endif