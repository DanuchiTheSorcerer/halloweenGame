#include "engine.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <vector>

using namespace nvcuda;
constexpr float pi = 3.14159265358979323846f; // Define pi as a constant
constexpr int maxAmountOfModels = 256*1024; // maximum amount of models device will store
constexpr int maxAmountOfTriangles = 64*1024*1024; // maximum amount of triangles that can be rendered
constexpr int maxAmountOfMeshes = 1024*1024; // maximum amount of triangles that can be rendered
Model* models; // device pointer to array of models
Triangle** triAllocs; // host pointer to array of dev ptrs to triangle arrays, obtained from allocation, for freeing
std::vector<int> usedModelIDs; // list of model ids allocated to, used for freeing
bool loadingModels; // meant to keep things synchronized when loading models
Triangle* scene; // array of triangles before rasterization
Mesh* meshes; // dev pointer to models for the tick
int meshCountThisTick;
Mesh* meshBuffer; // cpu copys to it (works like extension of interp buffer behavior)
int* meshSizes; // used for transformation outputs; ith index corresponds to the total number of triangles in the last i-1 meshes
float* triBright; //keep track of if a tri is culled (69420) and if not dot product of normal with camera (-1 to 1, brightness value) 
void clearModels() {
    for (int i = 0; i < usedModelIDs.size(); i++) {
        cudaFree(triAllocs[i]);
    }
    usedModelIDs.clear();
}
void loadModel(int ID, Triangle* triangles, int triangleCount) {
    if (loadingModels) {
        Model model;
        model.triangleCount = triangleCount;
        cudaMalloc(&model.triangles,sizeof(Triangle)*triangleCount);
        // copy model to device
        cudaMemcpy(&models[ID],&model,sizeof(Model),cudaMemcpyHostToDevice);
        // copy triangles to device
        cudaMemcpy(model.triangles,triangles,sizeof(Triangle)*triangleCount,cudaMemcpyHostToDevice);
        usedModelIDs.push_back(ID);
    }
};
void loadMesh(int modelID,float3 pos,float3 rotAxis, float scale, float theta) {
    Mesh mesh;
    mesh.modelID = modelID;
    mesh.pos = pos;
    mesh.rotAxis = rotAxis;
    mesh.scale = scale;
    mesh.theta = theta;

    cudaMemcpy(meshBuffer + meshCountThisTick,&mesh,sizeof(Mesh),cudaMemcpyHostToDevice);
    meshCountThisTick++;
}



struct Player {
    float3 position;
    float yaw;
    float walkSpeed;

};

Player player;
int2 lastMousePos;


interpolator tickLogic(int tickCount,int2 mouse,bool mousePressed,bool* keys) {
    interpolator result;
    meshCountThisTick = 0;
    if (tickCount == 0) {
        player.walkSpeed = 0.1f;
        player.yaw = pi;
        cudaMalloc(&scene,sizeof(Triangle) * maxAmountOfTriangles);
        cudaMalloc(&triBright,sizeof(float) * maxAmountOfTriangles);
        cudaMalloc(&meshes,sizeof(Mesh) * maxAmountOfMeshes);
        cudaMalloc(&meshBuffer,sizeof(Mesh) * maxAmountOfMeshes);
        cudaMalloc(&models, sizeof(Model) * maxAmountOfModels);
        cudaMalloc(&meshSizes,sizeof(int) * maxAmountOfMeshes);
        triAllocs = (Triangle**) malloc(sizeof(Triangle*) * maxAmountOfModels);
        loadingModels = true;
        Model cube;
        cube.triangles = new Triangle[12];

        float3 verts[8] = {
            {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
            {-1, -1,  1}, {1, -1,  1}, {1, 1,  1}, {-1, 1,  1}
        };

        // Define triangles (two per face)
        int faces[12][3] = {
            {2, 1, 0}, {3, 2, 0},  // Back face
            {6, 5, 1}, {2, 6, 1},  // Right face
            {7, 4, 5}, {6, 7, 5},  // Front face
            {3, 0, 4}, {7, 3, 4},  // Left face
            {6, 2, 3}, {7, 6, 3},  // Top face
            {1, 5, 4}, {0, 1, 4}   // Bottom face
        };

        for (int i = 0; i < 12; ++i) {
            cube.triangles[i].p1 = verts[faces[i][0]];
            cube.triangles[i].p2 = verts[faces[i][1]];
            cube.triangles[i].p3 = verts[faces[i][2]];
        }
        loadModel(0,cube.triangles,12);
    }
    if (tickCount == 200) {
        loadingModels = false;
    }
    if (!loadingModels) {
        if (keys[87]) {  // W - forward
            player.position.x+=(player.walkSpeed * sin(player.yaw));
            player.position.z+=(player.walkSpeed * cos(player.yaw));
        }
        if (keys[83]) {  // S - backward
            player.position.x-=(player.walkSpeed * sin(player.yaw));
            player.position.z-=(player.walkSpeed * cos(player.yaw));
        }
        if (keys[65]) {  // A - left
            player.position.x-=(player.walkSpeed * sin(player.yaw + pi/2));
            player.position.z-=(player.walkSpeed * cos(player.yaw + pi/2));
        }
        if (keys[68]) {  // D - right
            player.position.x+=(player.walkSpeed * sin(player.yaw + pi/2));
            player.position.z+=(player.walkSpeed * cos(player.yaw + pi/2));
        }
        if (mouse.x != 0) {
            player.yaw += (mouse.x) * 0.005f;
        }

        
        // Outer walls
        loadMesh(0, make_float3(0.0f, 0.0f, -5.0f), make_float3(0.0f, 1.0f, 0.0f), 5.0f, 0.0f);
        





    }

    if (true) {
        Camera camera;
        camera.pos = player.position;
        camera.rotAxis = make_float3(0.0f,1.0f,0.0f);
        camera.theta = player.yaw;
        camera.focalLength = 1.0f;
        result.tickCount = tickCount;
        result.models = models;
        result.loadingModels = loadingModels;
        result.camera = camera;
        result.scene = scene;
        result.meshBuffer = meshBuffer;
        result.meshes = meshes;
        result.bufferMeshCount = meshCountThisTick;
        result.meshSizes = meshSizes;
        result.triBright = triBright;
    }
    return result;
};
__global__ void computePixel(uint32_t* buffer, int width, int height, const interpolator* interp,float inpf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int speed = 3;
        int modVal = interp->tickCount*speed;
        uint32_t red   = 128 + 127*sinf((float)x/128 + ((float)modVal+inpf*speed)/60);
        uint32_t green = 128 + 127*cosf((float)y/128 + ((float)modVal+inpf*speed)/60);
        uint32_t blue  = 128 + 12*sinf((float)x/128 + (float)y/128 + ((float)modVal+inpf*speed)/6);
        buffer[idx] = 0xFF000000 | (red << 16) | (green << 8) | blue;
    }
}
__global__ void meshesTo2D(const interpolator* interp) { //fills scene with pre-rasterization triangles
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    

    __shared__ __align__(16) half M16[16][16]; // transforms point of the form (x,y,z,1) from model space to pre rasterize space

    // Only one thread per block needs to compute M16
    if (tid == 0) {

        float3 U1 = interp->meshes[bid].rotAxis;
        float THETA1 = interp->meshes[bid].theta;
        float S = interp->meshes[bid].scale;
        float3 T1 = interp->meshes[bid].pos;
        float3 T2 = interp->camera.pos;
        float3 U2 = interp->camera.rotAxis;
        float THETA2 = interp->camera.theta;
        float focalLength = interp->camera.focalLength;

        float M4[4][4];

        float invLen1 = rsqrtf(U1.x*U1.x + U1.y*U1.y + U1.z*U1.z);
        U1.x *= invLen1;  U1.y *= invLen1;  U1.z *= invLen1;
        float invLen2 = rsqrtf(U2.x*U2.x + U2.y*U2.y + U2.z*U2.z);
        U2.x *= invLen2;  U2.y *= invLen2;  U2.z *= invLen2;

        float c1 = cosf(THETA1), s1 = sinf(THETA1), C1 = 1.0f - c1;
        float R1[3][3] = {
            { c1 + U1.x*U1.x*C1,    U1.x*U1.y*C1 - U1.z*s1,  U1.x*U1.z*C1 + U1.y*s1 },
            { U1.y*U1.x*C1 + U1.z*s1, c1 + U1.y*U1.y*C1,     U1.y*U1.z*C1 - U1.x*s1 },
            { U1.z*U1.x*C1 - U1.y*s1, U1.z*U1.y*C1 + U1.x*s1, c1 + U1.z*U1.z*C1     }
        };

        float c2 = cosf(THETA2), s2 = -sinf(THETA2), C2 = 1.0f - c2;
        float R2[3][3] = {
            { c2 + U2.x*U2.x*C2,    U2.x*U2.y*C2 - U2.z*s2,  U2.x*U2.z*C2 + U2.y*s2 },
            { U2.y*U2.x*C2 + U2.z*s2, c2 + U2.y*U2.y*C2,     U2.y*U2.z*C2 - U2.x*s2 },
            { U2.z*U2.x*C2 - U2.y*s2, U2.z*U2.y*C2 + U2.x*s2, c2 + U2.z*U2.z*C2     }
        };

        float R[3][3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R[i][j] = R2[i][0]*R1[0][j]
                        + R2[i][1]*R1[1][j]
                        + R2[i][2]*R1[2][j];
            }
        }

        float3 RT1 = make_float3(
            R2[0][0]*T1.x + R2[0][1]*T1.y + R2[0][2]*T1.z,
            R2[1][0]*T1.x + R2[1][1]*T1.y + R2[1][2]*T1.z,
            R2[2][0]*T1.x + R2[2][1]*T1.y + R2[2][2]*T1.z
        );
        float3 RT2 = make_float3(
            R2[0][0]*T2.x + R2[0][1]*T2.y + R2[0][2]*T2.z,
            R2[1][0]*T2.x + R2[1][1]*T2.y + R2[1][2]*T2.z,
            R2[2][0]*T2.x + R2[2][1]*T2.y + R2[2][2]*T2.z
        );
        float3 t = make_float3(RT1.x - RT2.x,
                               RT1.y - RT2.y,
                               RT1.z - RT2.z);

        M4[0][0] = focalLength * S * R[0][0];
        M4[0][1] = focalLength * S * R[0][1];
        M4[0][2] = focalLength * S * R[0][2];
        M4[0][3] = focalLength *       t.x;

        M4[1][0] = focalLength * S * R[1][0];
        M4[1][1] = focalLength * S * R[1][1];
        M4[1][2] = focalLength * S * R[1][2];
        M4[1][3] = focalLength *       t.y;

        M4[2][0] =            S * R[2][0];
        M4[2][1] =            S * R[2][1];
        M4[2][2] =            S * R[2][2];
        M4[2][3] =                  t.z;

        for (int j = 0; j < 3; ++j) {
            M4[3][j] = 0.0f;
        }
        M4[3][3] = 1.0f;

        // Fill M16 with 4 M4s on its diagonal, 0s elsewhere
        // M16 is 16x16, M4 is 4x4
        for (int block = 0; block < 4; ++block) {
            for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int row = block * 4 + i;
                int col = block * 4 + j;
                M16[row][col] = __float2half(M4[i][j]);
            }
            }
        }
        // Set all other elements to 0
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
            // If not on one of the 4 M4 diagonal blocks, set to 0
            if (!((i / 4 == j / 4) && (i % 4 < 4) && (j % 4 < 4))) {
                M16[i][j] = __float2half(0.0f);
            }
            }
        }
    }

    // Make sure all threads see the finished M16
    __syncthreads();

    // ... now every thread in the block can use M16 ...

    //for all threads to use
    __shared__ __align__(16) half V16[16][16];
    __shared__ __align__(16) half V16_out[16][16];
    __shared__ Triangle* modTri;
    __shared__ int totalTriangles;

    if (tid == 0) {
        modTri = interp->models[interp->meshes[bid].modelID].triangles;
        totalTriangles = interp->models[interp->meshes[bid].modelID].triangleCount;
    }
    __syncthreads();
        
    // can only batch 20 triangles at a time, so loop until all vectors have been processed
    for (int batch = 0; batch < ((totalTriangles + 19) / 20); batch++) {
        //  ... STEPS ...
        // threads work together to fill V16
        // syncthreads
        // WMMA MatMul to produce transformed vectors
        // extract vectors 

        int trianglesThisBatch = min(20, totalTriangles - batch * 20); // 20 usually, less (or equal) on the last batch
        if (tid < trianglesThisBatch) {
            int TriIndex = tid + 20*batch; 
            float3 points[3];
            points[0] = modTri[TriIndex].p1;
            points[1] = modTri[TriIndex].p2;
            points[2] = modTri[TriIndex].p3;
            int row = tid / 5; 
            int column = tid %5;
            
            for (int i = 0;i < 3;i++) {
                V16[column*3 + i][row*4] = points[i].x;
                V16[column*3 + i][row*4 + 1] = points[i].y;
                V16[column*3 + i][row*4 + 2] = points[i].z;
                V16[column*3 + i][row*4 + 3] = 1;
            }

            
        }

        __syncthreads();

        //WMMA OPERATIONS HERE
        // multiply M16*V16, output to V16
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

        // Load A and B from shared memory
        wmma::load_matrix_sync(a_frag, &M16[0][0], 16);
        wmma::load_matrix_sync(b_frag, &V16[0][0], 16);

        // Initialize accumulator
        wmma::fill_fragment(c_frag, 0.0f);

        // Matrix Multiply-Accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        // Store result back into V16
        wmma::store_matrix_sync(&V16_out[0][0], c_frag, 16, wmma::mem_col_major);
        
        if (tid < trianglesThisBatch) {
            int TriIndex = tid + 20*batch;
            
            int row = tid / 5; 
            int column = tid %5; 
            interp->scene[interp->meshSizes[bid] + TriIndex].p1.x = V16_out[column*3 + 0][row*4 + 0];
            interp->scene[interp->meshSizes[bid] + TriIndex].p1.y = V16_out[column*3 + 0][row*4 + 1];
            interp->scene[interp->meshSizes[bid] + TriIndex].p1.z = V16_out[column*3 + 0][row*4 + 2];

            interp->scene[interp->meshSizes[bid] + TriIndex].p2.x = V16_out[column*3 + 1][row*4 + 0];
            interp->scene[interp->meshSizes[bid] + TriIndex].p2.y = V16_out[column*3 + 1][row*4 + 1];
            interp->scene[interp->meshSizes[bid] + TriIndex].p2.z = V16_out[column*3 + 1][row*4 + 2];

            interp->scene[interp->meshSizes[bid] + TriIndex].p3.x = V16_out[column*3 + 2][row*4 + 0];
            interp->scene[interp->meshSizes[bid] + TriIndex].p3.y = V16_out[column*3 + 2][row*4 + 1];
            interp->scene[interp->meshSizes[bid] + TriIndex].p3.z = V16_out[column*3 + 2][row*4 + 2];
        }
    }
}
__global__ void rasterizer(uint32_t* buffer, int width, int height, const interpolator* interp,float inpf) { // runs for each pixel, fills buffer 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    float wz = FLT_MAX;
    int color = 0;
    for (int i = 0; i < interp->meshSizes[interp->meshesCount]; i++) {
        float brightness = interp->triBright[i];
        if (interp->triBright[i] != 69420.0f) {
            Triangle* tri = &(interp->scene[i]);
            float3 p1 = tri->p1;
            float3 p2 = tri->p2;
            float3 p3 = tri->p3;
            float3 w = make_float3(((float) (2*x))/((float) width) - 1.0f,((float) (2*y))/((float) height) - 1.0f,0.0f);
            float3 u = make_float3(p2.x-p1.x,p2.y-p1.y,p2.z-p1.z);
            float3 v = make_float3(p3.x-p1.x,p3.y-p1.y,p3.z-p1.z);
            float3 q = make_float3(p3.x-p2.x,p3.y-p2.y,p3.z-p2.z);
            

            if (
                (((q.x*(w.y-p2.y) - q.y*(w.x-p2.x)) * (q.x*(p1.y-p2.y) - q.y*(p1.x-p2.x))) >= 0) && 
                (((v.x*(w.y-p1.y) - v.y*(w.x-p1.x)) * (v.x*u.y - v.y*u.x)) >= 0) && 
                (((u.x*(w.y-p1.y) - u.y*(w.x-p1.x)) * (u.x*v.y - u.y*v.x)) >= 0)
                ) {
                
                

                float denom = u.x*v.y - u.y*v.x;
float dx = w.x - p1.x;
float dy = w.y - p1.y;
w.z = p1.z + (u.z*(v.y*dx - v.x*dy) - v.z*(u.y*dx - u.x*dy)) / denom;


                if (w.z < wz) {
                    wz = w.z;
                    color = (int) (sqrt(brightness) * 255);
                }
            }
        }
    }

        buffer[idx] = 0xFF000000 | (color << 16) | (color << 8) | color;
}
__global__ void culler(const interpolator* interp, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
// Bounds check to prevent out-of-bounds access
    if (idx >= interp->meshSizes[interp->meshesCount]) {
        return;
    }
    Triangle* threadTri = &(interp->scene[idx]);

    // add in perspective
    if (threadTri->p1.z <= 0.0f && threadTri->p2.z <= 0.0f && threadTri->p3.z <= 0.0f) {
        interp->triBright[idx] = 69420.0f;
        return;
    }

    float3 v1 = make_float3(threadTri->p2.x - threadTri->p1.x,
                    threadTri->p2.y - threadTri->p1.y,
                    threadTri->p2.z - threadTri->p1.z);
        float3 v2 = make_float3(threadTri->p3.x - threadTri->p1.x,
                    threadTri->p3.y - threadTri->p1.y,
                    threadTri->p3.z - threadTri->p1.z);
        float3 bnormal = make_float3(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x
        );

    float aspect = (float)width / (float)height;

    threadTri->p1.x = (threadTri->p1.x / fabsf(threadTri->p1.z)) / aspect;
    threadTri->p1.y =  threadTri->p1.y / fabsf(threadTri->p1.z);

    threadTri->p2.x = (threadTri->p2.x / fabsf(threadTri->p2.z)) / aspect;
    threadTri->p2.y =  threadTri->p2.y / fabsf(threadTri->p2.z);

    threadTri->p3.x = (threadTri->p3.x / fabsf(threadTri->p3.z)) / aspect;
    threadTri->p3.y =  threadTri->p3.y / fabsf(threadTri->p3.z);

    v1 = make_float3(threadTri->p2.x - threadTri->p1.x,
                threadTri->p2.y - threadTri->p1.y,
                threadTri->p2.z - threadTri->p1.z);
    v2 = make_float3(threadTri->p3.x - threadTri->p1.x,
                threadTri->p3.y - threadTri->p1.y,
                threadTri->p3.z - threadTri->p1.z);
    float3 anormal = make_float3(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    );

    

    if ((threadTri->p1.x > 1.0f && threadTri->p2.x > 1.0f && threadTri->p3.x > 1.0f) || 
         (threadTri->p1.y > 1.0f && threadTri->p2.y > 1.0f && threadTri->p3.y > 1.0f) ||
         (threadTri->p1.x < -1.0f && threadTri->p2.x < -1.0f && threadTri->p3.x < -1.0f) || 
         (threadTri->p1.y < -1.0f && threadTri->p2.y < -1.0f && threadTri->p3.y < -1.0f)) {
        interp->triBright[idx] = 69420.0f;
    } else {
        //backface cull
        // Normalize normal
        float normLen = sqrtf(anormal.x * anormal.x + anormal.y * anormal.y + anormal.z * anormal.z);
        if (normLen > 0) {
            anormal.z /= normLen;
        }

        //camera facing 0,0,1


        if (anormal.z > 0) {
            interp->triBright[idx] = 69420.0f;
        } else {
            normLen = sqrtf(bnormal.x * bnormal.x + bnormal.y * bnormal.y + bnormal.z * bnormal.z);
            if (normLen > 0) {
                bnormal.z /= normLen;
            }
            interp->triBright[idx] =  -anormal.z;
        }


    }
}
__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor) {
    if (interp->loadingModels) {
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        computePixel<<<numBlocks, threadsPerBlock>>>(buffer, width, height, interp,interpolationFactor);
    } else {
        meshesTo2D<<<interp->meshesCount,32>>>(interp);

        __threadfence();

        int blocks = (interp->meshSizes[interp->meshesCount]+255)/256;

        culler<<<blocks,256>>>(interp, width, height);

        __threadfence();

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        rasterizer<<<numBlocks, threadsPerBlock>>>(buffer, width, height, interp,interpolationFactor);
    }
    
    __threadfence();
}
__device__ void interpolatorUpdateHandler(interpolator* interp) {
    if (interp->tickCount == 0) {
    }

    //copy buffer mesh to mesh
    for (int i = 0;i < interp->bufferMeshCount;i++) {
        *(interp->meshes + i) = *(interp->meshBuffer + i);
    }
    interp->meshesCount = interp->bufferMeshCount;

    //count mesh sizes
    int triTotals = 0;
    for (int i = 1; i < (interp->meshesCount + 1);i++) {
        triTotals = triTotals + interp->models[interp->meshes[i-1].modelID].triangleCount;
        interp->meshSizes[i] = triTotals;
    }
}
void cleanUpCall() {
    clearModels();
    cudaFree(models);
    cudaFree(scene);
    cudaFree(meshBuffer);
    cudaFree(meshes);
    cudaFree(meshSizes);
    cudaFree(triBright);
    free(triAllocs);
    usedModelIDs.clear();
}