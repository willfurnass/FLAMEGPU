#include "ORCA.h"
#include <cstdio>

//number of threads in a block - ideally 640. Must be a multiple of 32!
#define BlockDimSize 256
extern int SM_START;
extern int PADDING;
extern cudaStream_t stream1;
/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort = true)
{
    gpuAssert(cudaPeekAtLastError(), file, line);
#ifdef _DEBUG
    gpuAssert(cudaDeviceSynchronize(), file, line);
#endif

}

void stepORCA()
{
    pedestrian_make_orcaLines(stream1);
    cudaDeviceSynchronize();
    pedestrian_do_linear_program_2(stream1);
    cudaDeviceSynchronize();
    pedestrian_do_linear_program_3(stream1);
    cudaDeviceSynchronize();
}

extern int h_xmachine_memory_pedestrian_default_count;
extern int h_xmachine_memory_pedestrian_count;
extern __constant__ int d_xmachine_memory_pedestrian_count;
extern __constant__ int d_xmachine_memory_pedestrian_default_count;
extern xmachine_memory_pedestrian_list* h_pedestrians_default;
extern xmachine_memory_pedestrian_list* d_pedestrians_default;
extern xmachine_message_pedestrian_location_list* h_pedestrian_locations;
extern xmachine_message_pedestrian_location_list* d_pedestrian_locations;
extern xmachine_message_pedestrian_location_list* d_pedestrian_locations_swap;
extern int h_message_pedestrian_location_count;
extern int h_message_pedestrian_location_output_type;
#ifdef FAST_ATOMIC_SORTING
extern uint * d_xmachine_message_pedestrian_location_local_bin_index;
extern uint * d_xmachine_message_pedestrian_location_unsorted_index;
#else
extern uint * d_xmachine_message_pedestrian_location_keys;
extern uint * d_xmachine_message_pedestrian_location_values;
#endif
extern xmachine_message_pedestrian_location_PBM * d_pedestrian_location_partition_matrix;
extern glm::vec3 h_message_pedestrian_location_min_bounds;
extern glm::vec3 h_message_pedestrian_location_max_bounds;
extern glm::ivec3 h_message_pedestrian_location_partitionDim;
extern float h_message_pedestrian_location_radius;
extern int h_tex_xmachine_message_pedestrian_location_x_offset;
extern int h_tex_xmachine_message_pedestrian_location_y_offset;
extern int h_tex_xmachine_message_pedestrian_location_z_offset;
extern int h_tex_xmachine_message_pedestrian_location_vx_offset;
extern int h_tex_xmachine_message_pedestrian_location_vy_offset;
extern int h_tex_xmachine_message_pedestrian_location_pbm_start_offset;
extern int h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset;

/* Texture bindings */
/* pedestrian_location Message Bindings */
extern texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_x;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_x_offset;
extern texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_y;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_y_offset;
extern texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_z;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_z_offset;
extern texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_vx;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_vx_offset;
extern texture<float, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_vy;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_vy_offset;
extern texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_pbm_start;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_pbm_start_offset;
extern texture<int, 1, cudaReadModeElementType> tex_xmachine_message_pedestrian_location_pbm_end_or_count;
extern __constant__ int d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset;

//Environment Bounds
#define MIN_POSITION 0.8f
#define MAX_POSITION 300.0f

#define timeStep_ 0.8f
#define timeHorizon_ 5.0f
#define timeHorizonObst_ 5.0f
#define RVO_EPSILON 0.00001f
#define AGENTNO 128 //Adjust//Size of orcaLines max array size (same as within XMLmodel file array length) Max around 400 for all densities
#define maxSpeed_ 1.0f
#define RADIUS 0.5f
#define LOOKRADIUS 5.0f //equal to environment radius bin size.
#define NUMOBSLINES 0 //number of obstacle lines - not correctly implemented - keep 0

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Helper functions/////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Custom, unoptimized float atomic max
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, glm::floatBitsToInt(fmaxf(val, glm::intBitsToFloat(assumed))));
    } while (assumed != old);
    return glm::intBitsToFloat(old);
}

//Custom, unoptimized float atomic min
__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, glm::floatBitsToInt(fminf(val, glm::intBitsToFloat(assumed))));
    } while (assumed != old);
    return glm::intBitsToFloat(old);
}
//If agent is past bounds return a new random velocity vector back into the bounds
__FLAME_GPU_FUNC__ glm::vec2 isAtEdge(glm::vec2 agent_position, glm::vec2 agent_desv, RNG_rand48* rand48)
{
    float rand = rnd<CONTINUOUS>(rand48)*0.5f;

    //If the agent is close to the edge, give it a new random velocity
    if (agent_position.x < MIN_POSITION)
    {
        agent_desv.x = rand;
        //agent_desv.x = fabs(agent_desv.x);
        //agent_desv.x = 0.3f;
    }
    else if (agent_position.x > MAX_POSITION)
    {
        agent_desv.x = -rand;
        //agent_desv.x = -fabs(agent_desv.x);
        //agent_desv.x = -0.3f;
    }
    if (agent_position.y < MIN_POSITION)
    {
        agent_desv.y = rand;
        //agent_desv.y = fabs(agent_desv.y);
        //agent_desv.y = 0.3f;
    }
    else if (agent_position.y > MAX_POSITION)
    {
        agent_desv.y = -rand;
        //agent_desv.y = -fabs(agent_desv.y);
        //agent_desv.y = -0.3f;
    }


    return agent_desv;
}

//Keeps agents within bounds defined
__FLAME_GPU_FUNC__ glm::vec2 boundAgents(glm::vec2 agent_position)
{
    agent_position.x = (agent_position.x < MIN_POSITION) ? MAX_POSITION : agent_position.x;
    agent_position.x = (agent_position.x > MAX_POSITION) ? MIN_POSITION : agent_position.x;

    agent_position.y = (agent_position.y < MIN_POSITION) ? MAX_POSITION : agent_position.y;
    agent_position.y = (agent_position.y > MAX_POSITION) ? MIN_POSITION : agent_position.y;

    return agent_position;
}

//Determinant of 2 2d vectors
__FLAME_GPU_FUNC__ float det(const glm::vec2 v1, const glm::vec2 v2)
{
    return (v1.x * v2.y) - (v1.y * v2.x);
}

//Vector doted with itself
__FLAME_GPU_FUNC__ float absSq(const glm::vec2 &vector)
{
    return glm::dot(vector, vector);
}

//Squar of value a
__FLAME_GPU_FUNC__ float sqr(float a)
{
    return a * a;
}
/** \brief block level sum reduction. Result written to t0
* \param input_data thread value to be reduced over
* \returns sum reduction written to thread 0, else returns 0 for other threads.
*
* not valid for block size greater than 1024 (32*32)
*/
__device__ int reduce(int input_data) {
    __shared__ int s_ballot_results[BlockDimSize >> 5]; //shared results of the ballots
    int int_ret = 0; //value to return, only non zero for (threadIdx.x = 0)

    s_ballot_results[threadIdx.x >> 5] = ballot(input_data);
    __syncthreads();
    int blockCompNum;
    if (threadIdx.x < 32) { //0th warp and only threads that are within range - not valid for block size greater than 1024 (32*32)		
        if (threadIdx.x >= (BlockDimSize >> 5))
            blockCompNum = 0;
        else
            blockCompNum = __popc(s_ballot_results[threadIdx.x]);
        for (int offset = 16; offset>0; offset >>= 1)
            blockCompNum += __shfl_down(blockCompNum, offset);
    }
    if (threadIdx.x == 0)
        int_ret = blockCompNum;
    return int_ret;
}

/** \brief compresses the thread varaible input_data using warp shuffles
* \param input_data thread level
* \param compArr shared memory array to store the compressed indices
* \param comp_num the size of the compressed array
*/
__device__ int compress(int input_data, int *compArr) {

    const int tid = threadIdx.x;
    __shared__ int temp[BlockDimSize >> 5]; //stores warp scan results shared
    int int_ret; //value to return

    int temp1 = input_data;
    //scan within warp
    for (int d = 1; d<32; d <<= 1) {
        int temp2 = __shfl_up(temp1, d);
        if (tid % 32 >= d) temp1 += temp2;
    }
    if (tid % 32 == 31) temp[tid >> 5] = temp1;
    __syncthreads();
    //scan of warp sums
    if (threadIdx.x < 32) {
        int temp2 = 0.0f;
        if (tid < blockDim.x / 32)
            temp2 = temp[threadIdx.x];
        for (int d = 1; d<32; d <<= 1) {
            int temp3 = __shfl_up(temp2, d);
            if (tid % 32 >= d) temp2 += temp3;
        }
        if (tid < blockDim.x / 32) temp[tid] = temp2;
    }
    __syncthreads();
    //add to previous warp sums
    if (tid >= 32) temp1 += temp[tid / 32 - 1];
    //compress
    if (input_data == 1) {
        compArr[temp1 - 1] = threadIdx.x;
    }

    //get total number - reduction
    int_ret = reduce(input_data);

    return int_ret;
}
/* Shared memory size calculator for agent function */
int pedestrian_make_orcaLines_sm_size(int blockSize) {
	int sm_size;
	sm_size = SM_START;
	//Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_pedestrian_location));

	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);

	return sm_size;
}

/** pedestrian_make_orcaLines
* Agent function prototype for make_orcaLines function of pedestrian agent
*/
void pedestrian_make_orcaLines(cudaStream_t &stream) {

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


			//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
#if APPEND == 1
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;
#endif
	//set working count to current state count
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));



	//******************************** AGENT FUNCTION *******************************



	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_make_orcaLines, pedestrian_make_orcaLines_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_make_orcaLines_sm_size(blockSize);



	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)
	//any agent with discrete or partitioned message input uses texture caching
	size_t tex_xmachine_message_pedestrian_location_x_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_x_byte_offset, tex_xmachine_message_pedestrian_location_x, d_pedestrian_locations->x, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_x_offset = (int)tex_xmachine_message_pedestrian_location_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_x_offset, &h_tex_xmachine_message_pedestrian_location_x_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_y_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_y_byte_offset, tex_xmachine_message_pedestrian_location_y, d_pedestrian_locations->y, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_y_offset = (int)tex_xmachine_message_pedestrian_location_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_y_offset, &h_tex_xmachine_message_pedestrian_location_y_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_z_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_z_byte_offset, tex_xmachine_message_pedestrian_location_z, d_pedestrian_locations->z, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_z_offset = (int)tex_xmachine_message_pedestrian_location_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_z_offset, &h_tex_xmachine_message_pedestrian_location_z_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_vx_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_vx_byte_offset, tex_xmachine_message_pedestrian_location_vx, d_pedestrian_locations->vx, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_vx_offset = (int)tex_xmachine_message_pedestrian_location_vx_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_vx_offset, &h_tex_xmachine_message_pedestrian_location_vx_offset, sizeof(int)));
	size_t tex_xmachine_message_pedestrian_location_vy_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_vy_byte_offset, tex_xmachine_message_pedestrian_location_vy, d_pedestrian_locations->vy, sizeof(float)*xmachine_message_pedestrian_location_MAX));
	h_tex_xmachine_message_pedestrian_location_vy_offset = (int)tex_xmachine_message_pedestrian_location_vy_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_vy_offset, &h_tex_xmachine_message_pedestrian_location_vy_offset, sizeof(int)));
	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_pedestrian_location_pbm_start_byte_offset;
	size_t tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_pbm_start_byte_offset, tex_xmachine_message_pedestrian_location_pbm_start, d_pedestrian_location_partition_matrix->start, sizeof(int)*xmachine_message_pedestrian_location_grid_size));
	h_tex_xmachine_message_pedestrian_location_pbm_start_offset = (int)tex_xmachine_message_pedestrian_location_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_pbm_start_offset, &h_tex_xmachine_message_pedestrian_location_pbm_start_offset, sizeof(int)));
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset, tex_xmachine_message_pedestrian_location_pbm_end_or_count, d_pedestrian_location_partition_matrix->end_or_count, sizeof(int)*xmachine_message_pedestrian_location_grid_size));
	h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset = (int)tex_xmachine_message_pedestrian_location_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset, &h_tex_xmachine_message_pedestrian_location_pbm_end_or_count_offset, sizeof(int)));



	//MAIN XMACHINE FUNCTION CALL (make_orcaLines)
	//Reallocate   : false
	//Input        : pedestrian_location
	//Output       : 
	//Agent Output : 
#if APPEND == 1
	GPUFLAME_make_orcaLines << <g, b, sm_size, stream >> >(d_pedestrians, d_pedestrian_locations, d_pedestrian_location_partition_matrix);
#else
    GPUFLAME_make_orcaLines << <g, b, sm_size, stream >> >(d_pedestrians_default, d_pedestrian_locations, d_pedestrian_location_partition_matrix);
#endif
	gpuErrchkLaunch();


	//UNBIND MESSAGE INPUT VARIABLE TEXTURES
	//any agent with discrete or partitioned message input uses texture caching
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_x));
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_y));
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_z));
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_vx));
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_vy));
	//unbind pbm indices
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_pbm_start));
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_pedestrian_location_pbm_end_or_count));


	//************************ MOVE AGENTS TO NEXT STATE ****************************

	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count + h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX) {
		printf("Error: Buffer size of make_orcaLines agents in state default will be exceeded moving working agents to next state in function make_orcaLines\n");
		exit(0);
	}
#if APPEND == 1
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
#endif
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));


}




/* Shared memory size calculator for agent function */
int pedestrian_do_linear_program_2_sm_size(int blockSize) {
	int sm_size;
	sm_size = SM_START;

	return sm_size;
}

/* Shared memory size calculator for agent function */
int pedestrian_do_linear_program_2_WU_sm_size(int blockSize) {
	int sm_size;
	sm_size = SM_START;

	sm_size += sizeof(int) * blockSize * 2; //compArr + s_lineFail 
	sm_size += sizeof(int) * 2; //scanResult and active_agents
	sm_size += sizeof(glm::vec2) * blockSize * 2; //s_newv and s_desv
	sm_size += sizeof(int) * blockSize >> 5 * 2; //s_ballot_results + temp
	sm_size += sizeof(float4) * blockSize; // s_line
	sm_size += sizeof(float2) * blockSize; //s_t



	return sm_size;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Linear Program functions/////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Versions used agent memory

__FLAME_GPU_FUNC__ bool linearProgram1Disc(float radius, const glm::vec4 line, glm::vec2 *t)
{
    const glm::vec2 lines_direction_lineNo = glm::vec2(line.x, line.y);
    const glm::vec2 lines_point_lineNo = glm::vec2(line.z, line.w);

    const float dotProduct = glm::dot(lines_point_lineNo, lines_direction_lineNo);
    const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines_point_lineNo);

    if (discriminant < 0.0f) {
        // Max speed circle fully invalidates line lineNo. 
        return false;
    }

    float tLeft = -dotProduct - std::sqrt(discriminant);
    float tRight = -dotProduct + std::sqrt(discriminant);

    *t = glm::vec2(tLeft, tRight);

    return true;
}

__FLAME_GPU_FUNC__ bool linearProgram1Fractions(const glm::vec4 lines_lineNo, const glm::vec4 lines_i, const glm::vec2 t2, float* tnew, bool *tLeftb)
{
    //float tLeft = t2.x;//Unused
    //float tRight = t2.y;//Unused

    const glm::vec2 lines_direction_lineNo = glm::vec2(lines_lineNo.x, lines_lineNo.y);
    const glm::vec2 lines_point_lineNo = glm::vec2(lines_lineNo.z, lines_lineNo.w);


    const glm::vec2 lines_direction_i = glm::vec2(lines_i.x, lines_i.y);
    const glm::vec2 lines_point_i = glm::vec2(lines_i.z, lines_i.w);

    const float denominator = det(lines_direction_lineNo, lines_direction_i);
    const float numerator = det(lines_direction_i, lines_point_lineNo - lines_point_i);

    if (fabsf(denominator) <= RVO_EPSILON) {
        // Lines lineNo and i are (almost) parallel. 
        if (numerator < 0.0f) {
            return false;
        }
        else {
            //continue;
            *tnew = INT_MAX; //an arbitary large value  that is larger than tright
            *tLeftb = false;
            return true;
        }
    }

    const float t = numerator / denominator;

    if (denominator >= 0.0f) {
        // Line i bounds line lineNo on the right. 
        *tLeftb = false;
    }
    else {
        // Line i bounds line lineNo on the left. 
        *tLeftb = true;
    }
    *tnew = t;

    return true;
}

__FLAME_GPU_FUNC__ bool linearProgram1COMP(xmachine_memory_pedestrian* agent, int lineNo, float radius, const glm::vec2 &optVelocity, bool directionOpt, glm::vec2 &result, bool useProjLines, const glm::vec2 lines_direction_lineNo, const glm::vec2 lines_point_lineNo)
{
    /*glm::vec2 lines_direction_lineNo;
    glm::vec2 lines_point_lineNo;
    //if (useProjLines) {
    //	lines_direction_lineNo = glm::vec2(agent->projLine_direction_x[lineNo], agent->projLine_direction_y[lineNo]);
    //	lines_point_lineNo = glm::vec2(agent->projLine_point_x[lineNo], agent->projLine_point_y[lineNo]);
    //}
    //else {
    lines_direction_lineNo = glm::vec2(agent->orcaLine_direction_x[lineNo], agent->orcaLine_direction_y[lineNo]);
    lines_point_lineNo = glm::vec2(agent->orcaLine_point_x[lineNo], agent->orcaLine_point_y[lineNo]);
    //}*/

    const float dotProduct = glm::dot(lines_point_lineNo, lines_direction_lineNo);
    const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines_point_lineNo);

    if (discriminant < 0.0f) {
        // Max speed circle fully invalidates line lineNo. 
        return false;
    }

    float tLeft = -dotProduct - std::sqrt(discriminant);
    float tRight = -dotProduct + std::sqrt(discriminant);

    for (int i = 0; i < lineNo; ++i) {
        //glm::vec2 lines_direction_i = glm::vec2(agent->orcaLine_direction_x[i], agent->orcaLine_direction_y[i]);
        //glm::vec2 lines_point_i = glm::vec2agent->orcaLine_point_x[i], agent->orcaLine_point_y[i]);

        glm::vec4 temp = agent->orcaLine[i];
        const glm::vec2 lines_direction_i = glm::vec2(temp.x, temp.y);
        const glm::vec2 lines_point_i = glm::vec2(temp.z, temp.w);

        const float denominator = det(lines_direction_lineNo, lines_direction_i);
        const float numerator = det(lines_direction_i, lines_point_lineNo - lines_point_i);

        if (fabsf(denominator) <= RVO_EPSILON) {
            // Lines lineNo and i are (almost) parallel. 
            if (numerator < 0.0f) {
                return false;
            }
            else {
                continue;
            }
        }

        const float t = numerator / denominator;

        if (denominator >= 0.0f) {
            // Line i bounds line lineNo on the right. 
            tRight = fminf(tRight, t);
        }
        else {
            // Line i bounds line lineNo on the left. 
            tLeft = fmaxf(tLeft, t);
        }

        if (tLeft > tRight) {
            return false;
        }
    }

    if (directionOpt) {
        // Optimize direction. 
        if (glm::dot(optVelocity, lines_direction_lineNo) > 0.0f) {
            /* Take right extreme. */
            result = lines_point_lineNo + tRight * lines_direction_lineNo;
        }
        else {
            // Take left extreme. 
            result = lines_point_lineNo + tLeft * lines_direction_lineNo;
        }
    }
    else {
        // Optimize closest point. 
        const float t = glm::dot(lines_direction_lineNo, optVelocity - lines_point_lineNo);

        if (t < tLeft) {
            result = lines_point_lineNo + tLeft * lines_direction_lineNo;
        }
        else if (t > tRight) {
            result = lines_point_lineNo + tRight * lines_direction_lineNo;
        }
        else {
            result = lines_point_lineNo + t * lines_direction_lineNo;
        }
    }

    return true;
}
//Takes some linear program constraints and computes solution
__device__ void linear_program(const xmachine_memory_pedestrian_list * const agents, int thread_data, const int i, const glm::vec2 * const s_desv, glm::vec2 *s_newv, int *s_lineFail, const bool useProjLines) {
    //shared memory of compressed array
    __shared__ int compArr[BlockDimSize]; //shared compressed array working list
    __shared__ glm::vec4 s_line[BlockDimSize]; //shared current orca line of interest x:direction.x y:direction.y z:point.x w:point.y
    __shared__ int active_agents; //shared number of threads in block that are in the active compression
    __shared__ glm::vec2 s_t[BlockDimSize]; //tleft and tright shared

    //thread index	
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;

    //Calculate if agent needs to do work
    if (thread_data == 0) {
        if (useProjLines)
            s_line[tid] = agents->projLine[index + i * xmachine_memory_pedestrian_MAX];
        else
            s_line[tid] = agents->orcaLine[index + i * xmachine_memory_pedestrian_MAX];

        thread_data = (int)(det(glm::vec2(s_line[tid].x, s_line[tid].y), glm::vec2(s_line[tid].z, s_line[tid].w) - s_newv[tid]) > 0.0f);
    }
    else //thread not doing this calculation of i
        thread_data = 0;


    //compress through exclusive scan
    int result = compress(thread_data, compArr);
    //result written to thread0
    if (tid == 0)
        active_agents = result;

    //For compArr to be filled properly
    __syncthreads();


    //threads too high are ignored
    if (tid < active_agents) {
        //Get new index
        int n_tid = compArr[tid];

        //calculate tleft and tright and save in shared memory
        if (!linearProgram1Disc(maxSpeed_, s_line[n_tid], &s_t[n_tid])) {
            //failed here
            s_lineFail[n_tid] = i;
            //decrement the number of active agents
            //active_agents-- ?
        }
    }

    //Make sure SM is written to ok
    __syncthreads();

    //calculate the total number of work unit items (where a work unit is a line read and calculation for a unqiue agent line index). i.e. 
    int wu_count = (active_agents * i);

    //divide work unit items between threads. i.e.
    for (int j = 0; j < wu_count; j += BlockDimSize) {

        //calculate unique work unit index
        int wu_index = j + tid;

        //do work if there are still wu to complete
        if (wu_index < wu_count) {

            //for each thread work out which agent it is associated with
            int n_tid = compArr[wu_index / i];

            //for each thread work out which line index it should read
            int line_index = wu_index % i;

            //read in the unique agent line combination using the calculated indices
            glm::vec4 lines_i;
            if (useProjLines)
                lines_i = agents->projLine[n_tid + (blockIdx.x*blockDim.x) + (line_index*xmachine_memory_pedestrian_MAX)];
            else
                lines_i = agents->orcaLine[n_tid + (blockIdx.x*blockDim.x) + (line_index*xmachine_memory_pedestrian_MAX)];

            //calculate denominator and numerator			
            bool tleft;//whether the t value is left (or right)
            float t;//value of t
            if (!linearProgram1Fractions(s_line[n_tid], lines_i, s_t[n_tid], &t, &tleft)) {
                //operation failed
                s_lineFail[n_tid] = i;
            }

            //atomic write tleft and tright to shared memory using an atomic min and max
            if (tleft) {
                atomicMax(&s_t[n_tid].x, t);
            }
            else {
                atomicMin(&s_t[n_tid].y, t);
            }
        }
    }

    //sync to ensure all atomic writes are complete
    __syncthreads();

    //update the new velocity for each active agent
    if (tid < active_agents) {

        //Get new new index
        int n_tid = compArr[tid];

        //failure condition
        if (s_t[n_tid].x > s_t[n_tid].y)
            s_lineFail[n_tid] = i;

        //If not failed up to this point
        if (s_lineFail[n_tid] == -1) {

            // Optimize closest point 
            glm::vec2 lineDir = glm::vec2(s_line[n_tid].x, s_line[n_tid].y);
            glm::vec2 linePoint = glm::vec2(s_line[n_tid].z, s_line[n_tid].w);
            const float t = glm::dot(lineDir, s_desv[n_tid] - linePoint);

            if (t < s_t[n_tid].x) {
                s_newv[n_tid] = linePoint + s_t[n_tid].x * lineDir;
            }
            else if (t > s_t[n_tid].y) {
                s_newv[n_tid] = linePoint + s_t[n_tid].y * lineDir;
            }
            else {
                s_newv[n_tid] = linePoint + t * lineDir;
            }
        }
    }

    //sync to ensure all shared mem writes are complete
    __syncthreads();

}
//loops over all agent neighbours
__device__ int lp2(const xmachine_memory_pedestrian_list * const agents, const int count, const glm::vec2 optVelocity, const bool directionOpt, glm::vec2 &newVelocity, const bool useProjLines) {
    //shared memory of compressed array
    __shared__ int scanResult; //shared result of the scan
    __shared__ glm::vec2 s_newv[BlockDimSize]; //shared agents new speed after the calculation
    __shared__ glm::vec2 s_desv[BlockDimSize]; //shared desired speed
    __shared__ int s_lineFail[BlockDimSize]; //on which neighbour number the lp has failed shared

    //thread index	
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;

    //only update velocity if it needs to be done
    if (!(index >= d_xmachine_memory_pedestrian_count || count == 0)) {
        if (directionOpt) {
            // Optimize direction. Note that the optimization velocity is of unit length in this case.
            newVelocity = optVelocity * maxSpeed_;
        }
        else if (absSq(optVelocity) > sqr(maxSpeed_)) {
            // Optimize closest point and outside circle. 
            newVelocity = glm::normalize(optVelocity) * maxSpeed_;
        }
        else {
            // Optimize closest point and inside circle. 
            newVelocity = optVelocity;
        }
    }

    //store in shared memory
    s_newv[tid] = newVelocity;
    s_desv[tid] = optVelocity;
    s_lineFail[tid] = -1;

    //Loop over max possible size AGENTNO
    for (int i = 0; i < AGENTNO; i++) {


        //Update agent values each loop iteration
        //s_t[tid] = make_float2(0.f, 0.f);

        //early out - reduce over block. thread_data = 1 means not doing the calculation
        int thread_data = (int)(i >= count || s_lineFail[tid] != -1 || index >= d_xmachine_memory_pedestrian_count || count == 0);
        int aggregate = reduce(thread_data);
        //result written to thread0
        if (tid == 0)
            scanResult = (int)(aggregate == BlockDimSize);
        //Write to shared memory ok before accessing
        __syncthreads();
        if (scanResult)
            break;

        linear_program(agents, thread_data, i, s_desv, s_newv, s_lineFail, useProjLines);
    }

    newVelocity = s_newv[tid];

    return s_lineFail[tid];
}
__global__ void GPUFLAME_do_linear_program_2_WU(xmachine_memory_pedestrian_list* agents) {
    //thread index	
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    //const int tid = threadIdx.x;//unused

    //initialize values
    int count;
    glm::vec2 prefVelocity = glm::vec2(0, 0);
    glm::vec2 newVelocity = glm::vec2(0, 0);
    if (index < d_xmachine_memory_pedestrian_count) {
        count = agents->count[index];
        prefVelocity = glm::vec2(agents->desvx[index], agents->desvy[index]);
    }
    //run lp2
    int retVal = lp2(agents, count, prefVelocity, false, newVelocity, false);

    //save final values to global array
    if (index < d_xmachine_memory_pedestrian_count) {
        agents->vx[index] = newVelocity.x;
        agents->vy[index] = newVelocity.y;
        agents->lineFail[index] = retVal;
    }
}
/** pedestrian_do_linear_program_2
* Agent function prototype for do_linear_program_2 function of pedestrian agent
*/
void pedestrian_do_linear_program_2(cudaStream_t &stream) {

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


			//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
#if APPEND == 1
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;
#endif
	//set working count to current state count
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));



	//******************************** AGENT FUNCTION *******************************

	//whether to use the compression method for lp2 1= true
	//calculate the grid block size for main agent function
	//max grid size limited to 1024 due to reduction limits
	int blockSizeLimit = (state_list_size < 1024) ? state_list_size : 1024;
	gpuErrchk(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_do_linear_program_2_WU, pedestrian_do_linear_program_2_sm_size, blockSizeLimit));
	blockSize = BlockDimSize;
	sm_size = pedestrian_do_linear_program_2_sm_size(blockSize);

	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;



	//MAIN XMACHINE FUNCTION CALL (do_linear_program_2)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 

	//old verson with always appending
	////make a copy of the data for simulataneous testing
	//xmachine_memory_pedestrian_list* d_pedestrians2;
	//gpuErrchk(cudaMalloc((void**)&d_pedestrians2, sizeof(xmachine_memory_pedestrian_list)));
	//append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians2, d_pedestrians, 0, h_xmachine_memory_pedestrian_count);
	//gpuErrchkLaunch();

	GPUFLAME_do_linear_program_2_WU << <g, b, sm_size, stream >> >(d_pedestrians_default);

	//GPUFLAME_do_linear_program_2_COMP << <g, b, sm_size, stream >> >(d_pedestrians2);
	//cudaFree(d_pedestrians2);


	gpuErrchkLaunch();

	//************************ MOVE AGENTS TO NEXT STATE ****************************

	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count + h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX) {
		printf("Error: Buffer size of do_linear_program_2 agents in state default will be exceeded moving working agents to next state in function do_linear_program_2\n");
		exit(0);
	}
#if APPEND == 1
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
#endif
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));


}

/* Shared memory size calculator for agent function */
int pedestrian_do_linear_program_3_sm_size(int blockSize) {
	int sm_size;
	sm_size = SM_START;

	return sm_size;
}

//1D minimization to find closest possible solution of @optVelocity with computed orcaLines
//agent: pointer to agent of interest for calculations
//@lineNo: which line failed within linearProgram2
//@radius: maximum cutoff for consideration, usually set to agent's maximum velocity
//@optVelocity: the agents optimized velocity, usually set to its prefered veloicty. Saves a solution in @result that is the closest possible 
//@directionOpt: whether the optimized velocity needs normalizing
//@result: the agent's velocity at the end of the calculations
//@useProjLines: whether to use orcaLines or projLines (i.e. if running from makeOrcaLines or from LP3 function respectively)
//return -- false if failed to find solution, true otherwise
__FLAME_GPU_FUNC__ bool linearProgram1(xmachine_memory_pedestrian* agent, int lineNo, float radius, const glm::vec2 &optVelocity, bool directionOpt, glm::vec2 &result, bool useProjLines = false)
{
    glm::vec2 lines_direction_lineNo;
    glm::vec2 lines_point_lineNo;
    if (useProjLines) {
        //lines_direction_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_direction_x, lineNo), get_pedestrian_agent_array_value<float>(agent->projLine_direction_y, lineNo));
        //lines_point_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_point_x, lineNo), get_pedestrian_agent_array_value<float>(agent->projLine_point_y, lineNo));

        glm::vec4 temp = agent->projLine[lineNo];
        lines_direction_lineNo = glm::vec2(temp.x, temp.y);
        lines_point_lineNo = glm::vec2(temp.z, temp.w);
    }
    /*else {
    lines_direction_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, lineNo), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, lineNo));
    lines_point_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, lineNo), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, lineNo));
    }*/

    const float dotProduct = glm::dot(lines_point_lineNo, lines_direction_lineNo);
    const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines_point_lineNo);

    if (discriminant < 0.0f) {
        // Max speed circle fully invalidates line lineNo. 
        return false;
    }

    float tLeft = -dotProduct - std::sqrt(discriminant);
    float tRight = -dotProduct + std::sqrt(discriminant);

    for (int i = 0; i < lineNo; ++i) {
        glm::vec2 lines_direction_i;
        glm::vec2 lines_point_i;
        if (useProjLines) {
            //lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_direction_y, i));
            //lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_point_y, i));

            glm::vec4 temp = agent->projLine[i];
            lines_direction_i = glm::vec2(temp.x, temp.y);
            lines_point_i = glm::vec2(temp.z, temp.w);
        }
        /*else {
        lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, i));
        lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, i));
        }*/

        const float denominator = det(lines_direction_lineNo, lines_direction_i);
        const float numerator = det(lines_direction_i, lines_point_lineNo - lines_point_i);

        if (fabsf(denominator) <= RVO_EPSILON) {
            // Lines lineNo and i are (almost) parallel. 
            if (numerator < 0.0f) {
                return false;
            }
            else {
                continue;
            }
        }

        const float t = numerator / denominator;

        if (denominator >= 0.0f) {
            // Line i bounds line lineNo on the right. 
            tRight = fminf(tRight, t);
        }
        else {
            // Line i bounds line lineNo on the left. 
            tLeft = fmaxf(tLeft, t);
        }

        if (tLeft > tRight) {
            return false;
        }
    }

    if (directionOpt) {
        // Optimize direction. 
        if (glm::dot(optVelocity, lines_direction_lineNo) > 0.0f) {
            /* Take right extreme. */
            result = lines_point_lineNo + tRight * lines_direction_lineNo;
        }
        else {
            // Take left extreme. 
            result = lines_point_lineNo + tLeft * lines_direction_lineNo;
        }
    }
    else {
        // Optimize closest point. 
        const float t = glm::dot(lines_direction_lineNo, optVelocity - lines_point_lineNo);

        if (t < tLeft) {
            result = lines_point_lineNo + tLeft * lines_direction_lineNo;
        }
        else if (t > tRight) {
            result = lines_point_lineNo + tRight * lines_direction_lineNo;
        }
        else {
            result = lines_point_lineNo + t * lines_direction_lineNo;
        }
    }

    return true;
}
//Checks to see if current velocity satisfies the constrains of orcaLines. Calls |linearProgram1| if not satisfied
//@agent: pointer to agent of interest for calculations
//@count: actual number of elements within arrays to consider
//@radius: maximum cutoff for consideration, usually set to agent's maximum velocity
//@optVelocity: the agents optimized velocity, usually set to its prefered veloicty. Saves a solution in @result that is the closest possible 
//@directionOpt: whether the optimized velocity needs normalizing
//@result: the velocity at the end of the calculations
//@useProjLines: whether to use orcaLines or projLines (i.e. if running from makeOrcaLines or from LP3 function respectively)
//return -- count if solution is found. -1<i<count otherwise where i is the constraint that could not be solved
//template <bool useProjLines>
__FLAME_GPU_FUNC__ int linearProgram2(xmachine_memory_pedestrian* agent, const int count, float radius, const glm::vec2 optVelocity, bool directionOpt, glm::vec2 &result, bool useProjLines = false)
{
    if (directionOpt) {
        // Optimize direction. Note that the optimization velocity is of unit length in this case.
        result = optVelocity * radius;
    }
    else if (absSq(optVelocity) > sqr(radius)) {
        // Optimize closest point and outside circle. 
        result = glm::normalize(optVelocity) * radius;
    }
    else {
        // Optimize closest point and inside circle. 
        result = optVelocity;
    }
    //If first time calling lp2
    /*if (useProjLines == false) {
    for (int i = 0; i < count; ++i) {
    glm::vec2 lines_direction_i;
    glm::vec2 lines_point_i;

    lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, i));
    lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, i));

    if (det(lines_direction_i, lines_point_i - result) > 0.0f) {
    // Result does not satisfy constraint i. Compute new optimal result.
    const glm::vec2 tempResult = result;

    if (!linearProgram1(agent, i, radius, optVelocity, directionOpt, result, useProjLines)) {
    result = tempResult;
    return i;
    }
    }
    }
    }
    else*/ { //calling from lp3
        for (int i = 0; i < count; ++i) {
            glm::vec2 lines_direction_i;
            glm::vec2 lines_point_i;

            //lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_direction_y, i));
            //lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_point_y, i));

            glm::vec4 temp = agent->projLine[i];
            lines_direction_i = glm::vec2(temp.x, temp.y);
            lines_point_i = glm::vec2(temp.z, temp.w);

            if (det(lines_direction_i, lines_point_i - result) > 0.0f) {
                // Result does not satisfy constraint i. Compute new optimal result. 
                const glm::vec2 tempResult = result;

                if (!linearProgram1(agent, i, radius, optVelocity, directionOpt, result, useProjLines)) {
                    result = tempResult;
                    return i;
                }
            }
        }
    }

    return count;
}
//Finds best solution to satisfy constrants of orcaLines
//@agent: pointer to agent of interest for calculations
//@numObstLines: the number of orcaLines which correspond to obstacles (0 for me as no obstacles implemented)
//@beginLine: first line to not have solution from running linearProgram1 and 2
//@radius: maximum cutoff for consideration, usually set to agent's maximum velocity
//@result: the velocity at the end of the calculations
__FLAME_GPU_FUNC__ void linearProgram3(xmachine_memory_pedestrian* agent, int numObstLines, int beginLine, float radius, glm::vec2 &result)
{
    //const int index = (threadIdx.x + blockIdx.x*blockDim.x);//unused

    const int count = agent->count;
    float distance = 0.0f;

    for (int i = beginLine; i < count; ++i) {
        //glm::vec2 lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, i));
        //glm::vec2 lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, i));
        glm::vec4 temp = agent->orcaLine[i];
        glm::vec2 lines_direction_i = glm::vec2(temp.x, temp.y);
        glm::vec2 lines_point_i = glm::vec2(temp.z, temp.w);

        // Result does not satisfy constraint of line i.
        if (det(lines_direction_i, lines_point_i - result) > distance) {
            //number of elements within projLines arrays 						   
            int countlp3 = 0;
            for (int j = numObstLines; j < i; ++j) {

                //glm::vec2 lines_direction_j = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, j), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, j));
                //glm::vec2 lines_point_j = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, j), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, j));
                temp = agent->orcaLine[j];
                glm::vec2 lines_direction_j = glm::vec2(temp.x, temp.y);
                glm::vec2 lines_point_j = glm::vec2(temp.z, temp.w);

                glm::vec2 line_direction;
                glm::vec2 line_point;

                float determinant = det(lines_direction_i, lines_direction_j);

                if (fabsf(determinant) <= RVO_EPSILON) {
                    // Line i and line j are parallel. 
                    if (glm::dot(lines_direction_i, lines_direction_j) > 0.0f) {
                        // Line i and line j point in the same direction. 
                        continue;
                    }
                    else {
                        // Line i and line j point in opposite direction. 
                        line_point = 0.5f * (lines_point_i + lines_point_j);
                    }
                }
                else {
                    line_point = lines_point_i + (det(lines_direction_j, lines_point_i - lines_point_j) / determinant) * lines_direction_i;
                }

                line_direction = normalize(lines_direction_j - lines_direction_i);
                //Agent memory version
                //agent->projLine_direction_x[countlp3] = line_direction.x;
                //agent->projLine_direction_y[ountlp3] = line_direction.y;
                //agent->projLine_point_x[countlp3] = line_point.x;
                //agent->projLine_point_y[countlp3] = line_point.y;
                agent->projLine[countlp3] = glm::vec4(line_direction.x, line_direction.y, line_point.x, line_point.y);

                countlp3++;
            }

            /*if (index == 0) {
            printf("NOb i %i num %i, newVel: %f %f\n", i, index, result.x, result.y);
            for (int f = 0; f < i; f++) {
            printf("\t line %i x %f y %f z %f w %f\n", f, agent->projLine_direction_x[f], agent->projLine_direction_y[f], agent->projLine_point_x[f], agent->projLine_point_y[f]);
            }
            }*/
            const glm::vec2 tempResult = result;

            //If there is no solution found to the linear program minimization
            if (linearProgram2(agent, countlp3, radius, glm::vec2(-lines_direction_i.y, lines_direction_i.x), true, result, true) < countlp3) {
                // This should in principle not happen.  The result is by definition already in the feasible region of this linear program. If it fails, it is due to small floating point error, and the current result kept.
                //printf("Shouldn't happen\n");
                result = tempResult;
            }

            distance = det(lines_direction_i, lines_point_i - result);
            //if (index == 0)
            //	printf("NOa i %i num %i, newVel: %f %f\n", i, index, result.x, result.y);

        }
    }
}
// run linearProgram3 function if lineFail from agent memory is not equal to -1 having ran linearProgram 2
//@param agent Pointer to an agent structure of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
__FLAME_GPU_FUNC__ int do_linear_program_3(xmachine_memory_pedestrian* agent) {
    //Read in current new velocity from agent memory
    glm::vec2 newVelocity_ = glm::vec2(agent->vx, agent->vy);

    //Run lp3
    linearProgram3(agent, 0, agent->lineFail, maxSpeed_, newVelocity_);

    //write new velocity into agent memory
    agent->vx = newVelocity_.x;
    agent->vy = newVelocity_.y;

    return 0;
}
__global__ void GPUFLAME_do_linear_program_3(xmachine_memory_pedestrian_list* agents) {

    //continuous agent: index is agent position in 1D agent list
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_pedestrian_count)
        return;


    //SoA to AoS - xmachine_memory_do_linear_program_3 Coalesced memory read (arrays point to first item for agent index)
    xmachine_memory_pedestrian agent;
    agent.x = agents->x[index];
    agent.y = agents->y[index];
    agent.vx = agents->vx[index];
    agent.vy = agents->vy[index];
    agent.desvx = agents->desvx[index];
    agent.desvy = agents->desvy[index];
    agent.count = agents->count[index];
    agent.lineFail = agents->lineFail[index];
    //agent.newvx = agents->newvx[index];
    //agent.newvy = agents->newvy[index];
    agent.orcaLine = &(agents->orcaLine[index]);
    agent.projLine = &(agents->projLine[index]);

    //FLAME function call
    int dead = !do_linear_program_3(&agent);

    //continuous agent: set reallocation flag
    agents->_scan_input[index] = dead;

    //AoS to SoA - xmachine_memory_do_linear_program_3 Coalesced memory write (ignore arrays)
    agents->x[index] = agent.x;
    agents->y[index] = agent.y;
    agents->vx[index] = agent.vx;
    agents->vy[index] = agent.vy;
    agents->desvx[index] = agent.desvx;
    agents->desvy[index] = agent.desvy;
    agents->count[index] = agent.count;
    agents->lineFail[index] = agent.lineFail;
    //agents->newvx[index] = agent.newvx;
    //agents->newvy[index] = agent.newvy;
}
//optimised lp3 which calculates a valid velocity that penetrates within orcalines the least
#ifdef LP3COUNT
__global__ void GPUFLAME_do_linear_program_3_WU(xmachine_memory_pedestrian_list* agents, int * d_lp3Count, int iteration, int iterStart) {
#else
__global__ void GPUFLAME_do_linear_program_3_WU(xmachine_memory_pedestrian_list* agents) {
#endif
    //shared memories
    __shared__ int compArr[BlockDimSize]; //shared compressed array working list
    __shared__ glm::vec4 s_line[BlockDimSize]; //shared current orca line of interest x:direction.x y:direction.y z:point.x w:point.y
    __shared__ int active_agents; //shared number of threads in block that are in the active compression
    __shared__ int countlp3[BlockDimSize];
#ifdef LP3CUB
    __shared__ int loopMax;
    __shared__ int loopMin;
#endif

    //continuous agent: index is agent position in 1D agent list
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int tid = threadIdx.x;

    //velocity of agent after calculations
    glm::vec2 newVelocity_ = glm::vec2(agents->vx[index], agents->vy[index]);
    //least penetrative distance that satisfies constraints
    float distance = 0.0f;

    //#define LP3CUB //do cub min and max to find loop limits
#ifdef LP3CUB
    // Specialize BlockReduce for a 1D block of 128 threads on type int
    typedef cub::BlockReduce<int, BlockDimSize> BlockReduce;
    // Allocate shared memory for BlockReduce
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // Each thread obtains an input item
    int thread_data = agents->count[index];
    // Compute the block-wide max for thread0
    int max = BlockReduce(temp_storage).Reduce(thread_data, cub::Max());
    __syncthreads();
    // Each thread obtains an input item
    thread_data = agents->lineFail[index];
    // Compute the block-wide max for thread0
    int min = BlockReduce(temp_storage).Reduce(thread_data, cub::Min());
    if (tid == 0) {
        loopMax = max;
        loopMin = min;
        //printf("Max is %i \t min is %i\n", max, min);
    }
    __syncthreads();

    //can change 1 and AGENTNO
    for (int i = loopMin; i < loopMax; ++i) {
#else
    for (int i = 1; i < AGENTNO; ++i) {
#endif
        //Reset loop values
        countlp3[tid] = 0;

        //////////////////////////////////////
        //figure if agent will do work

        int thread_data = 0;
        if (index < d_xmachine_memory_pedestrian_count && i >= agents->lineFail[index] && i < agents->count[index] && agents->lineFail[index] != -1) {
            s_line[tid] = agents->orcaLine[index + i*xmachine_memory_pedestrian_MAX];
            glm::vec2 lines_direction_i = glm::vec2(s_line[tid].x, s_line[tid].y);
            glm::vec2 lines_point_i = glm::vec2(s_line[tid].z, s_line[tid].w);

            // Result does not satisfy constraint of line i - do work
            thread_data = (int)(det(lines_direction_i, lines_point_i - newVelocity_) > distance);
        }

        //create compressed work array
        int result = compress(thread_data, compArr);
        if (tid == 0) {
            active_agents = result;
#ifdef LP3COUNT
            if (iteration > iterStart) {
                atomicAdd(&d_lp3Count[(iteration - iterStart) * 128 + i], result);
            }
#endif
        }

        //Make sure SM is written to ok
        __syncthreads();

        //no work for this block for this neighbour value i
        if (active_agents == 0)
            continue;


        //////////////////////////////////////
        //Calculate the new set of orca lines using work units


        //calculate the total number of work unit items (where a work unit is a line read and calculation for a unqiue agent line index). i.e. 
        int wu_count = (active_agents * i);

        //divide work unit items between threads. i.e.
        for (int j = 0; j < wu_count; j += BlockDimSize) { //j=0 signifies no obstacle lines

            //calculate unique work unit index
            int wu_index = j + tid;

            //do work if there are still wu to complete
            if (wu_index < wu_count) {

                //for each thread work out which agent it is associated with
                const int n_tid = compArr[wu_index / i];
                const int newIndex = blockIdx.x * blockDim.x + n_tid;

                //for each thread work out which line index it should read
                const int line_index = wu_index % i;

                //read in the unique agent line combination using the calculated indices
                const glm::vec4 lines_i = agents->orcaLine[newIndex + line_index * xmachine_memory_pedestrian_MAX];
                const glm::vec2 lines_direction_j = glm::vec2(lines_i.x, lines_i.y);
                const glm::vec2 lines_point_j = glm::vec2(lines_i.z, lines_i.w);
                const glm::vec2 lines_direction_i = glm::vec2(s_line[n_tid].x, s_line[n_tid].y);
                const glm::vec2 lines_point_i = glm::vec2(s_line[n_tid].z, s_line[n_tid].w);
                glm::vec2 line_direction;
                glm::vec2 line_point;

                const float determinant = det(lines_direction_i, lines_direction_j);

                if (fabsf(determinant) <= RVO_EPSILON) {
                    // Line i and line j are parallel. 
                    if (glm::dot(lines_direction_i, lines_direction_j) > 0.0f) {
                        // Line i and line j point in the same direction. 
                        continue;
                    }
                    else {
                        // Line i and line j point in opposite direction. 
                        line_point = 0.5f * (lines_point_i + lines_point_j);
                    }
                }
                else {
                    line_point = lines_point_i + (det(lines_direction_j, lines_point_i - lines_point_j) / determinant) * lines_direction_i;
                }

                line_direction = normalize(lines_direction_j - lines_direction_i);

                //Agent memory version
                glm::vec4 temp = glm::vec4(line_direction, line_point);
                agents->projLine[newIndex + line_index*xmachine_memory_pedestrian_MAX] = temp;

                atomicAdd(&countlp3[n_tid], 1);
            }
        }

        __syncthreads();



        //////////////////////////////////////
        //Linear program up the new set of lines

        const glm::vec2 tempResult = newVelocity_;
        if (lp2(agents, countlp3[tid], glm::vec2(-s_line[tid].y, s_line[tid].x), true, newVelocity_, true) != -1) {
            // This should in principle not happen.  The result is by definition already in the feasible region of this linear program. If it fails, it is due to small floating point error, and the current result kept.
            //printf("Shouldn't happen\n");
            newVelocity_ = tempResult;
        }
        if (thread_data == 1) {
            distance = det(glm::vec2(s_line[tid].x, s_line[tid].y), glm::vec2(s_line[tid].z, s_line[tid].w) - newVelocity_);
        }
    }




    //Store values into global array structure
    if (index < d_xmachine_memory_pedestrian_count) {
        agents->vx[index] = newVelocity_.x;
        agents->vy[index] = newVelocity_.y;
    }
    }
/** pedestrian_do_linear_program_3
* Agent function prototype for do_linear_program_3 function of pedestrian agent
*/
void pedestrian_do_linear_program_3(cudaStream_t &stream) {
	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


			//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************
	//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
#if APPEND == 1
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;
#endif
	//set working count to current state count
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));





	//******************************** AGENT FUNCTION *******************************



	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_do_linear_program_3, pedestrian_do_linear_program_3_sm_size, state_list_size);
	blockSize = BlockDimSize;
	
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_do_linear_program_3_sm_size(blockSize);




	//MAIN XMACHINE FUNCTION CALL (do_linear_program_3)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	/*//make a copy of the data for simulataneous testing
	xmachine_memory_pedestrian_list* d_pedestrians2;
	gpuErrchk(cudaMalloc((void**)&d_pedestrians2, sizeof(xmachine_memory_pedestrian_list)));
	append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians2, d_pedestrians, 0, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();*/
#ifdef LP3COUNT
	GPUFLAME_do_linear_program_3_WU << <g, b, sm_size, stream >> >(d_pedestrians, d_lp3Count, iteration, iterStart);
#else
	GPUFLAME_do_linear_program_3_WU << <g, b, sm_size, stream >> > (d_pedestrians_default);
#endif
	gpuErrchkLaunch();

	/*GPUFLAME_do_linear_program_3 << <g, b, sm_size, stream >> >(d_pedestrians2);
	gpuErrchkLaunch();
	cudaFree(d_pedestrians2);*/


	//************************ MOVE AGENTS TO NEXT STATE ****************************

	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count + h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX) {
		printf("Error: Buffer size of do_linear_program_3 agents in state default will be exceeded moving working agents to next state in function do_linear_program_3\n");
		exit(0);
	}
#if APPEND == 1
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
#endif
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));

}

__global__ void GPUFLAME_make_orcaLines(xmachine_memory_pedestrian_list* agents, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix) {

    //continuous agent: index is agent position in 1D agent list
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    //For agents not using non partitioned message input check the agent bounds
    if (index >= d_xmachine_memory_pedestrian_count)
        return;


    //SoA to AoS - xmachine_memory_make_orcaLines Coalesced memory read (arrays point to first item for agent index)
    xmachine_memory_pedestrian agent;//Replace this with ORCA specific ped struct, save unnecessary data transfer
    agent.x = agents->x[index];
    agent.y = agents->y[index];
    agent.vx = agents->vx[index];
    agent.vy = agents->vy[index];
    agent.desvx = agents->desvx[index];
    agent.desvy = agents->desvy[index];
    agent.count = agents->count[index];
    agent.lineFail = agents->lineFail[index];
    //agent.newvx = agents->newvx[index];
    //agent.newvy = agents->newvy[index];
    agent.orcaLine = &(agents->orcaLine[index]);
    agent.projLine = &(agents->projLine[index]);

    //FLAME function call
    int dead = !make_orcaLines(&agent, pedestrian_location_messages, partition_matrix);

    //continuous agent: set reallocation flag
    agents->_scan_input[index] = dead;

    //AoS to SoA - xmachine_memory_make_orcaLines Coalesced memory write (ignore arrays)
    agents->x[index] = agent.x;
    agents->y[index] = agent.y;
    agents->vx[index] = agent.vx;
    agents->vy[index] = agent.vy;
    agents->desvx[index] = agent.desvx;
    agents->desvy[index] = agent.desvy;
    agents->count[index] = agent.count;
    agents->lineFail[index] = agent.lineFail;
    //agents->newvx[index] = agent.newvx;
    //agents->newvy[index] = agent.newvy;
}

__FLAME_GPU_FUNC__ int make_orcaLines(xmachine_memory_pedestrian* agent, xmachine_message_pedestrian_location_list* pedestrian_locations, xmachine_message_pedestrian_location_PBM* partition_matrix) {
    glm::vec2 position_ = glm::vec2(agent->x, agent->z);//z instead of y because 3d
    glm::vec2 velocity_ = glm::vec2(agent->vx, agent->vy);
    const glm::vec2 prefVelocity_ = glm::vec2(agent->desvx, agent->desvy);
    glm::vec2 newVelocity_ = glm::vec2(0.0, 0.0);
    const float radius_ = RADIUS;

    const float invTimeHorizon = 1.0f / timeHorizon_;

    int count = 0;

    // Create agent ORCA lines. 
    xmachine_message_pedestrian_location* current_message = get_first_pedestrian_location_message(pedestrian_locations, partition_matrix, agent->x, agent->y, agent->z);//Partitioned version
    //xmachine_message_pedestrian_location* current_message = get_first_pedestrian_location_message(pedestrian_locations); //BFversion
    while (current_message)
    {
        const glm::vec2 other_position_ = glm::vec2(current_message->x, current_message->y);
        const glm::vec2 other_velocity_ = glm::vec2(current_message->vx, current_message->vy);
        const float other_radius_ = RADIUS;


        const glm::vec2 relativePosition = other_position_ - position_;
        const glm::vec2 relativeVelocity = velocity_ - other_velocity_;
        const float distSq = absSq(relativePosition);
        const float combinedRadius = radius_ + other_radius_;
        const float combinedRadiusSq = sqr(combinedRadius);

        if ((glm::length(relativePosition) < LOOKRADIUS) && (position_ != other_position_)) // message outside the radius of interest and not the same message as agent
        {
            glm::vec2 line_direction = glm::vec2(0.0, 0.0);
            glm::vec2 line_point = glm::vec2(0.0, 0.0);
            glm::vec2 u = glm::vec2(0.0, 0.0);

            if (distSq >= combinedRadiusSq)
            {
                // No collision.
                const glm::vec2 w = relativeVelocity - invTimeHorizon * relativePosition;
                // Vector from cutoff center to relative velocity.
                const float wLengthSq = absSq(w);

                const float dotProduct1 = glm::dot(w, relativePosition);

                if (dotProduct1 < 0.0f && sqr(dotProduct1) > combinedRadiusSq * wLengthSq) {
                    // Project on cut-off circle.
                    const float wLength = std::sqrt(wLengthSq);
                    const glm::vec2 unitW = w / wLength;

                    line_direction = glm::vec2(unitW.y, -unitW.x);
                    u = (combinedRadius * invTimeHorizon - wLength) * unitW;
                }
                else
                {
                    // Project on legs.
                    const float leg = std::sqrt(distSq - combinedRadiusSq);

                    if (det(relativePosition, w) > 0.0f) {
                        // Project on left leg.
                        line_direction = glm::vec2(relativePosition.x * leg - relativePosition.y * combinedRadius, relativePosition.x * combinedRadius + relativePosition.y * leg) / distSq;
                    }
                    else {
                        // Project on right leg.
                        line_direction = -glm::vec2(relativePosition.x * leg + relativePosition.y * combinedRadius, -relativePosition.x * combinedRadius + relativePosition.y * leg) / distSq;
                    }

                    const float dotProduct2 = glm::dot(relativeVelocity, line_direction);

                    u = dotProduct2 * line_direction - relativeVelocity;
                }
            }
            else
            {
                // Collision. Project on cut-off circle of time timeStep.
                const float invTimeStep = 1.0f / timeStep_;

                // Vector from cutoff center to relative velocity.
                const glm::vec2 w = relativeVelocity - invTimeStep * relativePosition;

                const float wLength = glm::length(w);
                const glm::vec2 unitW = w / wLength;

                line_direction = glm::vec2(unitW.y, -unitW.x);
                u = (combinedRadius * invTimeStep - wLength) * unitW;

                //If deep within eachother collision, report it. Cannot be exactly equal to radius due to floating point errors calling false-positive collisions
                if (distSq < 0.49990*0.49990)
                {
                    //	printf("Collision at x:%f y:%f!\n", agent->x, agent->y);
                }
            }

            line_point = velocity_ + (0.5f * u);

            //Set the values into the array
            glm::vec4 temp = glm::vec4(line_direction.x, line_direction.y, line_point.x, line_point.y);
            agent->orcaLine[count] = temp;


            //Move onto next message
            count++;
            if (count >= AGENTNO) {
                count = AGENTNO - 1;
                printf("warning: More agents than allowed for in max size at x:%f y:%f\n", agent->x, agent->y);
            }

        }
        current_message = get_next_pedestrian_location_message(current_message, pedestrian_locations, partition_matrix);

    }
    agent->count = count;


    return 0;
}