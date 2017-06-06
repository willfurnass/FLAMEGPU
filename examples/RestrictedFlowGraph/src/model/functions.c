
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

//@todo introduce on_vertex and on_edge as states instead of flags? would output to seperate messaage lists.

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include <header.h>

#define ON_VERTEX 0
#define ON_EDGE 1

 // Declare global scope variables for host-based agent creation, so allocation of host data is only performed once.
xmachine_memory_Agent ** h_agent_AoS;
//const unsigned int h_agent_AoS_MAX = xmachine_memory_Agent_MAX;
const unsigned int h_agent_AoS_MAX = 8;

__FLAME_GPU_INIT_FUNC__ void initialiseHost() {
	unsigned int max_lifespan = 16;
	set_MAX_LIFESPAN(&max_lifespan);

	//@todo use better rng.
	// Seed the host random number generator.
	srand(0);

	// Allocate host Aos
	h_agent_AoS = h_allocate_agent_Agent_array(h_agent_AoS_MAX);
	
}
__FLAME_GPU_INIT_FUNC__ void generateAgentInit() {

	// For upto the defined number of agents, randomly distribute agents throughout the network.
	for (unsigned int i = 0; i < h_agent_AoS_MAX; i++) {
		xmachine_memory_Agent * agent = h_agent_AoS[i];
		agent->id = i;
		agent->current_graph_element_id = i % 2; //@todo
		agent->next_graph_element_id = (i+1) % 2; //@todo
		agent->next_graph_element_population = 0;
		agent->vertex_or_edge = ON_VERTEX; // @todo ON_EDGE
		agent->has_intent = false;
		agent->x = 0.0f; //@todo
		agent->y = 0.0f; //@todo
		agent->z = 0.0f; //@todo
	}

	h_add_agents_Agent_default(h_agent_AoS, h_agent_AoS_MAX);


}
__FLAME_GPU_EXIT_FUNC__ void freeHostMemory() {

	// Free host allocated memory
	h_free_agent_Agent_array(&h_agent_AoS, h_agent_AoS_MAX);


}
__FLAME_GPU_STEP_FUNC__ void generateAgentStep() {
	//@todo - Do I need this?

}

/**
 * output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int output_location(xmachine_memory_Agent* agent, xmachine_message_location_list* location_messages, RNG_rand48* rand48){
	// @todo - select next edge/vertex.
	agent->next_graph_element_id = (agent->id + 1) % 2; //@todo

	// Output location as message.
	add_location_message(
		location_messages, 
		agent->id, 
		agent->current_graph_element_id, 
		agent->vertex_or_edge, 
		agent->x, 
		agent->y, 
		agent->z
	);
    return 0;
}

/**
 * read_locations FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param location_messages  location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.
 */
__FLAME_GPU_FUNC__ int read_locations(xmachine_memory_Agent* agent, xmachine_message_location_list* location_messages){

	// Get the next elements capcity from the graph data structure @todo
	unsigned int target_element_capacity = 10; //@todo
	int target_vertex_or_edge = (agent->vertex_or_edge == ON_VERTEX) ? ON_EDGE : ON_VERTEX;

	// Local counter for the number of agent on the target edge.
	unsigned int target_element_population = 0;
	
    
	// For each location message
    xmachine_message_location* current_message = get_first_location_message(location_messages);
    while (current_message)
    {
        // Count the number of messages for my target edge.
		if (current_message->current_graph_element_id == agent->next_graph_element_id && current_message->vertex_or_edge == target_vertex_or_edge) {
			target_element_population += 1;
		}
        current_message = get_next_location_message(current_message, location_messages);
    }    
  
	// If the element has capacity, remember this.
	if (target_element_population < target_element_capacity) {
		// Save the element population for later.
		//@todo store remaining capcaity instead?
		agent->next_graph_element_population = target_element_population;
		// Set the intent flag.
		agent->has_intent = true;
	}
	else {
		agent->next_graph_element_population = 0;
		agent->has_intent = false;
	}

    return 0;
}

/**
 * declare_intent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param intent_messages Pointer to output message list of type xmachine_message_intent_list. Must be passed as an argument to the add_intent_message function ??.
 */
__FLAME_GPU_FUNC__ int declare_intent(xmachine_memory_Agent* agent, xmachine_message_intent_list* intent_messages){

    
    /* 
    //Template for message output function use unsigned int id = 0;unsigned int current_graph_element_id = 0;int vertex_or_edge = 0;float x = 0;float y = 0;float z = 0;
    add_intent_message(intent_messages, id, next_graph_element_id, vertex_or_edge, x, y, z);
    */
	int target_vertex_or_edge = (agent->vertex_or_edge == ON_VERTEX) ? ON_EDGE : ON_VERTEX;

	add_intent_message(intent_messages, agent->id, agent->next_graph_element_id, target_vertex_or_edge, agent->x, agent->y, agent->z);
    
  
    return 0;
}

/**
 * resolve_intent FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_Agent. This represents a single agent instance and can be modified directly.
 * @param intent_messages  intent_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_intent_message and get_next_intent_message functions.
 */
__FLAME_GPU_FUNC__ int resolve_intent(xmachine_memory_Agent* agent, xmachine_message_intent_list* intent_messages){

	// Count the number of agent attempting to move to the same target location, and track the number of agents with lower ID than me. 
	// Get the capacity
	unsigned int target_element_capacity = 10; //@todo
	int target_vertex_or_edge = (agent->vertex_or_edge == ON_VERTEX) ? ON_EDGE : ON_VERTEX;
	unsigned int target_element_intent_count = 0;
	unsigned int lower_id_count = 0;


    xmachine_message_intent* current_message = get_first_intent_message(intent_messages);
    while (current_message)
    {
		if (current_message->next_graph_element_id == agent->next_graph_element_id && current_message->vertex_or_edge == target_vertex_or_edge) {
			target_element_intent_count++;
			if (current_message->id < agent->id) {
				lower_id_count++;
			}
		}
        
        current_message = get_next_intent_message(current_message, intent_messages);
    }
     
	// Calc the remaining capcity.
	unsigned int remaining_capcaity = target_element_capacity - agent->next_graph_element_population;
	// If there is sufficient capcaity for the agents before me and then me, we can move!
	if (remaining_capcaity - lower_id_count > 1) {
		printf("Agent %u moved from %u to %u\n", agent->id, agent->current_graph_element_id, agent->next_graph_element_id);
		agent->current_graph_element_id = agent->next_graph_element_id;
	}
	// reset the intent flag
	agent->has_intent = false;
	//@todo reset other values?

    return 0;
}

  


#endif //_FLAMEGPU_FUNCTIONS
