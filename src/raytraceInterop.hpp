#ifndef __RAYTRACE_INTEROP_H__
#define __RAYTRACE_INTEROP_H__

#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <ctime>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector_types.h>

#include "misc.h"
#include "common_defines.h"

namespace RaytraceInterop {

extern "C"
__global__
void raytrace(cudaSurfaceObject_t image, const float* miscData, float* deviceRes);

//
GLuint viewGLTexture;
cudaGraphicsResource_t viewCudaResource;
//
int mouseOldX = 0;
int mouseOldY = 0;
int mouseX    = 0;
int mouseY    = 0;

//hostArr holds misc params for the CUDA kernel
float* hostArr   = NULL;
//same as hostArr, but points to the device memory
float* deviceArr = NULL;
//holds the final buffer on the device
float* deviceRes = NULL;

//for marking when a key has been pressed
bool keyStates[256] = {false};

void initGL() {
    int argc = 0; char** argv = NULL;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("");
    
    glewInit();
    
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void initCUDA() {
    check(cudaGLSetGLDevice(0));
    check(cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    
    //allocate misc params array and final buffer array
    if (hostArr == NULL) {
        hostArr = new float[PARAM_LAST];
        
        for (int i = 0; i < PARAM_LAST; ++i) {
            hostArr[i] = 0.f;
        }
        
        hostArr[PARAM_WIDTH]  = WIDTH;
        hostArr[PARAM_HEIGHT] = HEIGHT;
        hostArr[PARAM_DEPTH]  = 1;
        
        check(cudaMalloc((void**)&deviceArr, PARAM_LAST * sizeof(float)));
        
        check(cudaMemcpy(deviceArr, hostArr, sizeof(float) * PARAM_LAST, cudaMemcpyHostToDevice));
        
        check(cudaMalloc((void**)&deviceRes, WIDTH * HEIGHT * 4 * sizeof(float)));
    }
}

//resets the buffer calculated so far
void restart() {
    hostArr[PARAM_ITER]=0;
    srand(0);
}

//inits everything nessecary
void init() {
    initGL();
    initCUDA();
    restart();
}

//cleans up eveyrthing nessecary
void deinit() {
    delete[] hostArr;
    hostArr = NULL;
    
    check(cudaFree(deviceArr));
    check(cudaFree(deviceRes));

    check(cudaDeviceReset());
}

//calls cuda kernel; openGL texture is ready to be shown directly after this has been called
void callCUDAKernel(cudaSurfaceObject_t image) {
    dim3 block;
    block.x = 8;
    block.y = 8;
    
    dim3 grid;
    grid.x = sqrt(WIDTH * HEIGHT / (block.x * block.y));
    grid.y = grid.x;
    hostArr[PARAM_ITER]++;
    hostArr[PARAM_RAND]  = float(rand()) / float(RAND_MAX);
    hostArr[PARAM_RAND2] = float(rand()) / float(RAND_MAX);
    
    check(cudaMemcpy(deviceArr, hostArr, sizeof(float)*PARAM_LAST, cudaMemcpyHostToDevice));
    
    raytrace<<<grid, block>>>(image, deviceArr, deviceRes);
    check(cudaPeekAtLastError());
    check(cudaDeviceSynchronize());
}

//updates misc params if any key has been pressed
void checkKey(unsigned char key){
    if (keyStates[key]!=true)
        return;
    
    float* miscData = hostArr;
    switch (key){
            //esc
        case (27) :
            exit(EXIT_SUCCESS);
            break;
        case 'f':
            miscData[PARAM_SPHERE1_LEFT_RIGHT] -= 1;
            break;
        case 'h':
            miscData[PARAM_SPHERE1_LEFT_RIGHT] += 1;
            break;
        case 't':
            miscData[PARAM_SPHERE1_FRONT_BACK] -= 1;
            break;
        case 'g':
            miscData[PARAM_SPHERE1_FRONT_BACK] += 1;
            break;
        case 'w':
            miscData[PARAM_SPHERE2_FRONT_BACK] -= 1;
            break;
        case 'a':
            miscData[PARAM_SPHERE2_LEFT_RIGHT] -= 1;
            break;
        case '1':
            miscData[PARAM_DEPTH] += 1;
            break;
        case '2':
            miscData[PARAM_DEPTH] -= 1;
            break;
        case 's':
            miscData[PARAM_SPHERE2_FRONT_BACK] += 1;
            break;
        case 'd':
            miscData[PARAM_SPHERE2_LEFT_RIGHT] += 1;
            break;
        case 'r':
            miscData[PARAM_SPHERE1_UP_DOWN] += 1;
            break;
        case 'y':
            miscData[PARAM_SPHERE1_UP_DOWN] -= 1;
            break;
        case 'q':
            miscData[PARAM_SPHERE2_UP_DOWN] -= 1;
            break;
        case 'e':
            miscData[PARAM_SPHERE2_UP_DOWN] += 1;
            break;
        default:
            break;
    }
    restart();
}

void renderFrameProc() {
    //for monitoring fps
    static unsigned long long fps = 0;
    static unsigned long long frameNumber = 0;
    
    ++frameNumber;

    //check if any key has been pressed
    for (int i = 0; i < 256; ++i)
        checkKey((unsigned char)(i));
    
    //get what time it is
    clock_t time = clock();
    
    //***********
    //call CUDA kernel to make 1 raytrace pass
    check(cudaGraphicsMapResources(1, &viewCudaResource));
    cudaArray_t viewCudaArray;
    check(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));
    cudaResourceDesc viewCudaArrayResourceDesc;
    memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
    viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
    cudaSurfaceObject_t viewCudaSurfaceObject;
    check(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
    
    callCUDAKernel(viewCudaSurfaceObject);
    
    check(cudaDestroySurfaceObject(viewCudaSurfaceObject));
    check(cudaGraphicsUnmapResources(1, &viewCudaResource));
    check(cudaStreamSynchronize(0));
    
    //***********
    //draw the interop buffer with OpenGL
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    {
        glBegin(GL_QUADS);
        {
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, +1.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, +1.0f);
        }
        glEnd();
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
    
    //***********
    //calc the fps
    const int currentFPS = int(1.0 / diffclock(clock(), time));
    fps += currentFPS;
    
    //update the window title with fps info every 42 frames.
    if (frameNumber % 42 == 0) {
        char title[32];
        memset(title, 0, sizeof(char) * 32);
        sprintf(title, "FPS : %i", int(fps/frameNumber));
        glutSetWindowTitle(title);
    }
    
    //request new frame
    glutPostRedisplay();
}

//this mouse procs are little buggy
void mouseProc(int button, int state, int x, int y){
    mouseOldX = x;
    mouseOldY = y;
}

void motionProc(int x, int y) {
    mouseX = x;
    mouseY = y;
    //update misc params with mouse pos (this moves the camera)
    //divide by 500 to make camera moves slower
    hostArr[PARAM_MOUSE_LEFT_RIGHT] -= (mouseOldX - mouseX) / 500.f;
    hostArr[PARAM_MOUSE_UP_DOWN]    -= (mouseOldY - mouseY) / 500.f;
    
    restart();
}

void keyDownProc(unsigned char key, int x, int y) {
    keyStates[key] = true;
}

void keyUpProc(unsigned char key, int x, int y){
    keyStates[key] = false;
}

}//namespace RaytraceInterop
int main(int argc, char **argv) {
    RaytraceInterop::init();
    
    glutDisplayFunc(RaytraceInterop::renderFrameProc);
    glutKeyboardFunc(RaytraceInterop::keyDownProc);
    glutKeyboardUpFunc(RaytraceInterop::keyUpProc);
    glutMouseFunc(RaytraceInterop::mouseProc);
    glutMotionFunc(RaytraceInterop::motionProc);
    glutMainLoop();
    
    RaytraceInterop::deinit();
}


#endif //__RAYTRACE_INTEROP_H__