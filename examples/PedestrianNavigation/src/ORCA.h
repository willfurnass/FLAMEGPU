#ifndef __ORCA_h__
#define __ORCA_h__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>
#include "header.h"

void pedestrian_make_orcaLines(cudaStream_t &stream);
int pedestrian_do_linear_program_2_sm_size(int blockSize);
int pedestrian_do_linear_program_2_WU_sm_size(int blockSize);
void pedestrian_do_linear_program_2(cudaStream_t &stream);
int pedestrian_do_linear_program_3_sm_size(int blockSize);
void pedestrian_do_linear_program_3(cudaStream_t &stream);
__global__ void GPUFLAME_make_orcaLines(xmachine_memory_pedestrian_list* agents, xmachine_message_pedestrian_location_list* properties_message_messages, xmachine_message_pedestrian_location_PBM* partition_matrix);
__FLAME_GPU_FUNC__ int make_orcaLines(xmachine_memory_pedestrian* agent, xmachine_message_pedestrian_location_list* properties_messages, xmachine_message_pedestrian_location_PBM* partition_matrix);
#endif //__ORCA_h__