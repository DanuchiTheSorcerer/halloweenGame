#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// A simple structure holding game state for interpolation.
// Extend this structure with more state as needed.

struct Camera {
    float3 pos;
    float3 rotAxis;
    float theta;
    float focalLength;
};

struct Triangle {
    float3 p1;
    float3 p2;
    float3 p3;
};

struct Model {
    Triangle* triangles;
    int triangleCount;
};

struct Mesh {
    int modelID;
    float scale;
    float3 pos;
    float3 rotAxis;
    float theta;
};

struct interpolator  {
    int tickCount;
    Model* models;
    bool loadingModels;
    Camera camera;
    Triangle* scene;
    Mesh* meshBuffer;
    Mesh* meshes;
    int bufferMeshCount;
    int meshesCount;
    int* meshSizes;
    float* triBright;
};


// tickLogic
// Processes game logic for the given tick count and returns an interpolator 
// containing the updated game state.
interpolator tickLogic(int tickCount,int2 mouse,bool mousePressed,bool* keys);

//compute a frame
__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor);

//put gpu response to new interpolator
__device__ void interpolatorUpdateHandler(interpolator* interp);

//clean up for use by game engine user
void cleanUpCall();

#ifdef __cplusplus
}
#endif

#endif ENGINE_H
