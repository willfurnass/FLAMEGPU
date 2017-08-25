
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>


/**
 * output_example FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages Pointer to output message list of type xmachine_message_example_message_list. Must be passed as an argument to the add_example_message_message function ??.
 */

__FLAME_GPU_FUNC__ int output_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages){
    
    //Template for message output function use 
    //
    //
    if(agent->id == 0){
        add_example_message_message(example_message_messages, agent->id, agent->x, agent->y, agent->z);
        agent->id = -1;
    }
     
     
     
  
    return 0;
}

/**
 * input_example FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_example_agent. This represents a single agent instance and can be modified directly.
 * @param example_message_messages  example_message_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_example_message_message and get_next_example_message_message functions.
 */
#if defined(xmachine_message_example_message_grid_size)
__FLAME_GPU_FUNC__ int input_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages, xmachine_message_example_message_PBM* partition_matrix){
#else
__FLAME_GPU_FUNC__ int input_example(xmachine_memory_example_agent* agent, xmachine_message_example_message_list* example_message_messages){
#endif

    int tid = threadIdx.x + (blockDim.x * blockIdx.x);
    if(tid == 0){
        printf("d_message_example_message_count: %d\n", d_message_example_message_count);
    }
    
    #if defined(xmachine_message_example_message_grid_size)
    xmachine_message_example_message* current_message = get_first_example_message_message(example_message_messages, partition_matrix, agent->x, agent->y, agent->z);
    #else
    xmachine_message_example_message* current_message = get_first_example_message_message(example_message_messages);
    #endif
    while (current_message)
    {
        if(tid == 0){
            printf("agent %d: message from %d\n", agent->id, current_message->id);
        }
        
        #if defined(xmachine_message_example_message_grid_size)
        current_message = get_next_example_message_message(current_message, example_message_messages, partition_matrix);
        #else
        current_message = get_next_example_message_message(current_message, example_message_messages);
        #endif
    }
     
     
    
  
    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
