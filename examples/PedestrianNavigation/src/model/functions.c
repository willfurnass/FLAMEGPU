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

#include <random>
#include "GlobalsController.h"
#include "header.h"
#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

//Cuda call
static void HandleCUDAError(const char *file,
    int line,
    cudaError_t status = cudaGetLastError()) {
#ifdef _DEBUG
    cudaDeviceSynchronize();
#endif
    if (status != CUDA_SUCCESS || (status = cudaGetLastError()) != CUDA_SUCCESS)
    {
        if (status == cudaErrorUnknown)
        {
            printf("%s(%i) An Unknown CUDA Error Occurred :(\n", file, line);
            printf("Perhaps performing the same operation under the CUDA debugger with Memory Checker enabled could help!\n");
            printf("If this error only occurs outside of NSight debugging sessions, or causes the system to lock up. It may be caused by not passing the required amount of shared memory to a kernal launch that uses runtime sized shared memory.\n", file, line);
            printf("Also possible you have forgotten to allocate texture memory you are trying to read\n");
            printf("Passing a buffer to 'cudaGraphicsSubResourceGetMappedArray' or a texture to 'cudaGraphicsResourceGetMappedPointer'.\n");
            getchar();
            exit(1);
        }
        printf("%s(%i) CUDA Error Occurred;\n%s\n", file, line, cudaGetErrorString(status));
        getchar();
        exit(1);
    }
}
#define CUDA_CALL( err ) (HandleCUDAError(__FILE__, __LINE__ , err))
#define CUDA_CHECK() (HandleCUDAError(__FILE__, __LINE__))

#include "Graph.h"
#include "glm/gtx/component_wise.hpp"
#define SCALE_FACTOR 0.03125

#define I_SCALER (SCALE_FACTOR*0.35f)
#define MESSAGE_RADIUS d_message_pedestrian_location_radius
#define MIN_DISTANCE 0.0001f
__constant__ float HUMAN_HEIGHT;
__constant__ float NEXT_EDGE_THRESHOLD;

//#define NUM_EXITS 7

#define PI 3.1415f
#define RADIANS(x) (PI / 180.0f) * x

__device__ Graph d_nav;
Graph h_nav;

std::default_random_engine rng;
std::uniform_real_distribution<float> floatDist(0,1);
extern void rescalePedestrianPop(float newHeight);//PedestrianPopulation.cpp
extern void rescaleNavMapPopulation(float scaleFactor, glm::vec3 offset);//NavMapPopulation.cpp
std::string navPath;
void stepORCA();
void setNavPath(const char *_navPath)
{
    navPath = _navPath;
}
__FLAME_GPU_INIT_FUNC__ void loadGraph()
{
    void *ptr;
    h_nav.load(navPath.c_str());
    assert(h_nav.vertex.entryCount > 0);
    //Scale the locations to FLAME range min(-1) max(1)
    //Find location min and max
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    glm::vec3 min2 = glm::vec3(FLT_MAX);
    glm::vec3 max2 = glm::vec3(-FLT_MAX);
    for (unsigned int i = 0; i < h_nav.point.count; ++i)
    {//only compare points, vertex loc's are duplication of this
        min = glm::min(min, h_nav.point.loc[i]);
        max = glm::max(max, h_nav.point.loc[i]);
    }
    glm::vec3 scaleFactor = 2.0f/(max - min);
    float scaleFactor1D = glm::min(scaleFactor.x, scaleFactor.z);
    scaleFactor = glm::vec3(scaleFactor1D);//Uniform scaling
    //To scale we multiply by 'scaleFactor' and subtractMin
    glm::vec3 centerY = glm::vec3(0, 1.0f - (((max.y - min.y)*scaleFactor1D) / 2.0f), 0);//This extra offset centers the model on the Y axis, this makes it work better with FLAME's camera
    for (unsigned int i = 0; i < h_nav.point.count; ++i)
    {
        h_nav.point.loc[i] = ((h_nav.point.loc[i] - min) *scaleFactor) - glm::vec3(1.0f) + centerY;
        min2 = glm::min(min2, h_nav.point.loc[i]);
        max2 = glm::max(max2, h_nav.point.loc[i]);
    }
    for (unsigned int i = 0; i < h_nav.vertex.count; ++i)
    {
        h_nav.vertex.loc1[i] = ((h_nav.vertex.loc1[i] - min) *scaleFactor) - glm::vec3(1.0f) + centerY;
        h_nav.vertex.loc2[i] = ((h_nav.vertex.loc2[i] - min) *scaleFactor) - glm::vec3(1.0f) + centerY;
    }
    max = (max*scaleFactor) - glm::vec3(1.0f);
    printf("max(%.3f, %.3f, %.3f)\n", max.x, max.y, max.z);
    printf("max2(%.3f, %.3f, %.3f)\n", max2.x, max2.y, max2.z);
    //Setup human height on device
    float humanHeight = 1.8288f*scaleFactor1D;//6ft in metres * scale factor
    float nextEdgeThreshold = 0.005*scaleFactor1D;//10cm * scale factor//Why does this need to be so small???
    rescalePedestrianPop(humanHeight);
    rescaleNavMapPopulation(scaleFactor1D, -(min + glm::vec3(1.0f / scaleFactor1D) - (centerY/scaleFactor1D)));//Include offset for moving to -1,+1 range
    CUDA_CALL(cudaMemcpyToSymbol(HUMAN_HEIGHT, &humanHeight, sizeof(float)));
    CUDA_CALL(cudaMemcpyToSymbol(NEXT_EDGE_THRESHOLD, &nextEdgeThreshold, sizeof(float)));
    //Copy to device
    Graph *dp_nav;
    CUDA_CALL(cudaGetSymbolAddress((void**)&dp_nav, d_nav));
    //Vertex
    CUDA_CALL(cudaMemcpy(&dp_nav->vertex.count, &h_nav.vertex.count, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(&dp_nav->vertex.entryCount, &h_nav.vertex.entryCount, sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(glm::vec3)*h_nav.vertex.count));
    CUDA_CALL(cudaMemcpy(&dp_nav->vertex.loc1, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.vertex.loc1, sizeof(glm::vec3)*h_nav.vertex.count, cudaMemcpyHostToDevice));
    
    CUDA_CALL(cudaMalloc(&ptr, sizeof(glm::vec3)*h_nav.vertex.count));
    CUDA_CALL(cudaMemcpy(&dp_nav->vertex.loc2, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.vertex.loc2, sizeof(glm::vec3)*h_nav.vertex.count, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(unsigned int*)*h_nav.vertex.entryCount));
    CUDA_CALL(cudaMemcpy(&dp_nav->vertex.routes, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    for (unsigned int i = 0; i<h_nav.vertex.entryCount; ++i)
    {
        void *ptr2;
        CUDA_CALL(cudaMalloc(&ptr2, sizeof(unsigned int)*h_nav.vertex.count));
        CUDA_CALL(cudaMemcpy(((void**)ptr) + i, &ptr2, sizeof(void*), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(ptr2, h_nav.vertex.routes[i], sizeof(unsigned int)*h_nav.vertex.count, cudaMemcpyHostToDevice));
    }

    CUDA_CALL(cudaMalloc(&ptr, sizeof(unsigned int)*(h_nav.vertex.count + 1)));
    CUDA_CALL(cudaMemcpy(&dp_nav->vertex.first_edge_index, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.vertex.first_edge_index, sizeof(unsigned int)*(h_nav.vertex.count + 1), cudaMemcpyHostToDevice));

    //Edge
    CUDA_CALL(cudaMemcpy(&dp_nav->edge.count, &h_nav.edge.count, sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(unsigned int)*h_nav.edge.count));
    CUDA_CALL(cudaMemcpy(&dp_nav->edge.source, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.edge.source, sizeof(unsigned int)*h_nav.edge.count, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(unsigned int)*h_nav.edge.count));
    CUDA_CALL(cudaMemcpy(&dp_nav->edge.destination, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.edge.destination, sizeof(unsigned int)*h_nav.edge.count, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(unsigned int)*h_nav.edge.count));
    CUDA_CALL(cudaMemcpy(&dp_nav->edge.poly, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.edge.poly, sizeof(unsigned int)*h_nav.edge.count, cudaMemcpyHostToDevice));

    //Poly
    CUDA_CALL(cudaMemcpy(&dp_nav->poly.count, &h_nav.poly.count, sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(unsigned int)*(h_nav.poly.count + 1)));
    CUDA_CALL(cudaMemcpy(&dp_nav->poly.first_point_index, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.poly.first_point_index, sizeof(unsigned int)*(h_nav.poly.count + 1), cudaMemcpyHostToDevice));

    //Point
    CUDA_CALL(cudaMemcpy(&dp_nav->point.count, &h_nav.point.count, sizeof(unsigned int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&ptr, sizeof(glm::vec3)*h_nav.point.count));
    CUDA_CALL(cudaMemcpy(&dp_nav->point.loc, &ptr, sizeof(void*), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(ptr, h_nav.point.loc, sizeof(glm::vec3)*h_nav.point.count, cudaMemcpyHostToDevice));
}
__FLAME_GPU_EXIT_FUNC__ void freeGraph()
{
    void *ptr;
    Graph *dp_nav;
    CUDA_CALL(cudaGetSymbolAddress((void**)&dp_nav, d_nav));
    //Vertex
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->vertex.loc1, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->vertex.loc2, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));
    unsigned int entryCount = 0;
    CUDA_CALL(cudaMemcpy(&entryCount, &dp_nav->vertex.loc2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    for (unsigned int i = 0; i<entryCount; ++i)
    {
        CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->vertex.routes + i, sizeof(void*), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(ptr));
    }
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->vertex.first_edge_index, sizeof(void*), cudaMemcpyDeviceToHost));

    //Edge
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->edge.source, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->edge.destination, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->edge.poly, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));

    //Poly
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->poly.first_point_index, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));

    //Point
    CUDA_CALL(cudaMemcpy(&ptr, &dp_nav->point.loc, sizeof(void*), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(ptr));

    CUDA_CALL(cudaFree(ptr));
    h_nav.free();
}
extern int h_xmachine_memory_pedestrian_count;
__FLAME_GPU_STEP_FUNC__ void spawnAgents()
{
    if (h_xmachine_memory_pedestrian_count >= 5000)
        return;
    int exitState[7] = { getStateExit1(), getStateExit2(), getStateExit3(), getStateExit4(), getStateExit5(), getStateExit6(), getStateExit7() };
    float emissionRate[7] = { getEmmisionRateExit1(), getEmmisionRateExit2(), getEmmisionRateExit3(), getEmmisionRateExit4(), getEmmisionRateExit5(), getEmmisionRateExit6(), getEmmisionRateExit7() };
    float exitProbabilitities[7] = { getProbabilityExit1(), getProbabilityExit2(), getProbabilityExit3(), getProbabilityExit4(), getProbabilityExit5(), getProbabilityExit6(), getProbabilityExit7() };
    //Accumulate probabilities
    exitProbabilitities[0] *= exitState[0];
    for (unsigned int i = 1; i < 7; ++i)
        exitProbabilitities[i] = (exitState[i] * exitProbabilitities[i] * (int)(i < h_nav.vertex.entryCount)) + exitProbabilitities[i - 1];
    std::uniform_real_distribution<float> exitDist(0, exitProbabilitities[6]);
    //Clear probabilty for closed exits
    for (unsigned int i = 1; i < 7; ++i)
        exitProbabilitities[i] = (exitState[i] * exitProbabilitities[i] * (int)(i < h_nav.vertex.entryCount));
    //Attempt to spawn at each exit
    static int qt = 0;
    for (unsigned int i = 0; i < h_nav.vertex.entryCount&&i<7;++i)
    {
        //If exit is open
        if (exitState[i] && floatDist(rng)<emissionRate[i])
        {
            //Select an exit
            float exitPart = exitDist(rng);
            unsigned int exit = 0;
            for (; exit < h_nav.vertex.entryCount; ++exit)
            {
                if (exitPart < exitProbabilitities[exit] && exitProbabilitities[exit]!=0.0f)
                    break;
            }
            if (exit == h_nav.vertex.entryCount || exit == i)
                continue;//No exits open?

            //Calc agent location
            glm::vec3 loc = glm::mix(h_nav.vertex.loc1[i], h_nav.vertex.loc2[i], 0.5);
            //Create agent
            xmachine_memory_pedestrian agent;
            agent.animate = floatDist(rng);
            agent.speed = floatDist(rng)*0.5f + 1.0f;
            agent.animate_dir = 1.0f;
            agent.current_edge = h_nav.vertex.routes[exit][i];
            agent.next_edge = h_nav.vertex.routes[exit][h_nav.edge.destination[agent.current_edge]];
            agent.exit_no = exit;
            agent.count = 0;
            //agent.orcaLine = glm::vec4(0);//Agent array, not var
            //agent.projLine = glm::vec4(0);//Agent array, not var
            agent.lod = 1.0f;
            agent.x = loc.x;
            agent.y = loc.y;
            agent.z = loc.z;
            agent.desvx = 0.0f;
            agent.desvy = 0.0f;
            agent.vx = 0.0f;
            agent.vy = 0.0f;
            //printf("Agent(Entry: %d, Exit: %d, Loc(%.3f, %.3f, %.3f), curr: %d, next: %d\n", i, exit, loc.x, loc.y, loc.z, agent.current_edge, agent.next_edge);
            h_add_agent_pedestrian_default(&agent);
        }
    }
}

__FLAME_GPU_FUNC__ int getNewExitLocation(RNG_rand48* rand48){

    unsigned int exitCt = d_nav.vertex.entryCount;
	int exit1_compare = EXIT1_PROBABILITY;
    int exit2_compare = exitCt <= 2 ? (EXIT2_PROBABILITY + exit1_compare) : 0;
    int exit3_compare = exitCt <= 3 ? (EXIT3_PROBABILITY + exit2_compare) : 0;
    int exit4_compare = exitCt <= 4 ? (EXIT4_PROBABILITY + exit3_compare) : 0;
    int exit5_compare = exitCt <= 5 ? (EXIT5_PROBABILITY + exit4_compare) : 0;
    int exit6_compare = exitCt <= 6 ? (EXIT6_PROBABILITY + exit5_compare) : 0;
    int exit7_compare = exitCt <= 7 ? (EXIT7_PROBABILITY + exit6_compare) : 0;

    float rand = rnd<DISCRETE_2D>(rand48)*exit7_compare;

	if (rand<exit1_compare)
		return 0;
	else if (rand<exit2_compare)
		return 1;
	else if (rand<exit3_compare)
		return 2;
	else if (rand<exit4_compare)
		return 3;
	else if (rand<exit5_compare)
		return 4;
	else if (rand<exit6_compare)
		return 5;
    else if (rand < exit7_compare)
        return 6;
    else
        return 0;//This shouldn't occur?

}
__FLAME_GPU_FUNC__ bool exitIsOpen(int exitId)
{
    if (exitId >= d_nav.vertex.entryCount)
        return false;

    switch (exitId)
    {
    case 0:
        return EXIT1_STATE;
    case 1:
        return EXIT2_STATE;
    case 2:
        return EXIT3_STATE;
    case 3:
        return EXIT4_STATE;
    case 4:
        return EXIT5_STATE;
    case 5:
        return EXIT6_STATE;
    case 6:
        return EXIT7_STATE;
    default:
        return false;//If it's an invalid exit, treat as closed
    }

}
/**
 * output_location FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_pedestrian_location(xmachine_memory_pedestrian* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages){

    
    add_pedestrian_location_message(pedestrian_location_messages, agent->x, agent->y, agent->z, agent->vx, agent->vy);
  
    return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
 * @param pedestrian_location_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_location_message and get_next_location_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_location_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.* @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
/*__FLAME_GPU_FUNC__ int avoid_pedestrians(xmachine_memory_pedestrian* agent, xmachine_message_pedestrian_location_list* pedestrian_location_messages, xmachine_message_pedestrian_location_PBM* partition_matrix, RNG_rand48* rand48){

	glm::vec2 agent_pos = glm::vec2(agent->x, agent->z);
	glm::vec2 agent_vel = glm::vec2(agent->vx, agent->vy);

	glm::vec2 navigate_velocity = glm::vec2(0.0f, 0.0f);
	glm::vec2 avoid_velocity = glm::vec2(0.0f, 0.0f);

    xmachine_message_pedestrian_location* current_message = get_first_pedestrian_location_message(pedestrian_location_messages, partition_matrix, agent->x, agent->y, agent->z);
	while (current_message)
	{
		glm::vec2 message_pos = glm::vec2(current_message->x, current_message->z);
		float separation = length(agent_pos - message_pos);
        float ySep = abs(current_message->y - agent->y);
        if ((separation < MESSAGE_RADIUS) && (separation>MIN_DISTANCE) && (ySep<HUMAN_HEIGHT)){
			glm::vec2 to_agent = normalize(agent_pos - message_pos);
			float ang = acosf(dot(agent_vel, to_agent));
			float perception = 45.0f;

			//STEER
			if ((ang < RADIANS(perception)) || (ang > 3.14159265f-RADIANS(perception))){
				glm::vec2 s_velocity = to_agent;
				s_velocity *= powf(I_SCALER/separation, 1.25f)*STEER_WEIGHT;
				navigate_velocity += s_velocity;
			}

			//AVOID
			glm::vec2 a_velocity = to_agent;
			a_velocity *= powf(I_SCALER/separation, 2.00f)*AVOID_WEIGHT;
			avoid_velocity += a_velocity;						

		}
		 current_message = get_next_pedestrian_location_message(current_message, pedestrian_location_messages, partition_matrix);
	}

	//maximum velocity rule
	glm::vec2 steer_velocity = navigate_velocity + avoid_velocity;

	agent->desvx = steer_velocity.x;
	agent->desvy = steer_velocity.y;

    return 0;
}*/

__device__ bool isCW(const glm::vec2* p0, const glm::vec2* p1, const glm::vec2* p2)
{
    return (p1->x*p2->y - p1->y*p2->x - p0->x*p2->y + p0->y*p2->x + p0->x*p1->y - p0->y*p1->x)< 0;
}
__device__ bool insideConvexPoly2D(unsigned int beforeFirstPoint, unsigned int afterLastPoint, const glm::vec2 *testPoint)
{
    glm::vec2 pt1;
    glm::vec2 pt2 = glm::vec2(d_nav.point.loc[afterLastPoint - 1]);
    for (unsigned int i = beforeFirstPoint; i < afterLastPoint; ++i)
    {
        pt1 = pt2;
        pt2 = glm::vec2(d_nav.point.loc[i]);
        if (!isCW(&pt1, &pt2, testPoint))
        {
            return false;
        }
    }
    return true;
}
//Projects the 3d poly into the 2d plane of Normal(0,1,0) and ensures point is in bounds
__device__ bool insideConvexPoly3D2D(unsigned int beforeFirstPoint, unsigned int afterLastPoint, const glm::vec3 *testPoint)
{
    glm::vec2 testPoint2D = glm::vec2(testPoint->x, testPoint->z);
    glm::vec2 pt1;
    glm::vec2 pt2 = glm::vec2(d_nav.point.loc[afterLastPoint - 1].x, d_nav.point.loc[afterLastPoint - 1].z);
    float yMin = FLT_MAX;
    float yMax = -FLT_MAX;
    for (unsigned int i = beforeFirstPoint; i < afterLastPoint; ++i)
    {
        pt1 = pt2;
        pt2 = glm::vec2(d_nav.point.loc[i].x, d_nav.point.loc[i].z);
        if (!isCW(&pt1, &pt2, &testPoint2D))
        {
            return false;
        }
        yMin = glm::min(yMin, d_nav.point.loc[afterLastPoint - 1].y);
        yMax = glm::max(yMax, d_nav.point.loc[afterLastPoint - 1].y);
    }
    if (testPoint->y >= yMin&&testPoint->y <= yMax)
        return true;
    return false;
}
__device__ bool insideConvexPoly3D(unsigned int beforeFirstPoint, unsigned int afterLastPoint, const glm::vec3 *testPoint)
{
    //This technique is probably expensive to perform on the GPU as it involves transforming the polygon
    //https://github.com/juj/MathGeoLib/blob/master/src/Geometry/Polygon.cpp#L361
    return false;
}
__device__ float distPointLine(const glm::vec3 &line1, const glm::vec3 &line2, const glm::vec3 &point)
{
    //return abs(((line2.y-line1.y)*point.x)-((line2.x-line1.x)*point.y)+(line2.x*line1.y)-(line2.y-line1.x))/distance(line1, line2);//2D
    {//http://www.randygaul.net/2014/07/23/distance-point-to-line-segment/
        glm::vec3 n = line2 - line1;
        glm::vec3 pa = line1 - point;
        float c = dot(n, pa);
        if (c > 0.0f)//Nearest point goes past A
            return dot(pa,pa);
        glm::vec3 bp = point - line2;
        if (dot(n, bp) > 0.0f)//Nearest point goes past B
            return dot(bp,bp);
        glm::vec3 e = pa - n * (c / dot(n, n));
        return dot(e, e);
    }

    //return length(cross(line2 - line1, line1 - point)) / length(line2 - line1);//Infinite
}
__device__ glm::vec3 closestPointonLineSegment(const glm::vec3 &line1, const glm::vec3 &line2, const glm::vec3 &point)
{
    glm::vec3 line = line2 - line1;
    float len = length(line);
    line = normalize(line);
    glm::vec3 v = point - line1;
    float d = dot(v, line);
    d = glm::clamp<float>(d, 0, len);
    return line1 + line*d;
}
/**
* force_flow FLAMEGPU Agent Function
* Automatically generated using functions.xslt
* @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly.
*/
__FLAME_GPU_FUNC__ int force_flow(xmachine_memory_pedestrian* agent, RNG_rand48* rand48)
{
    assert(agent->current_edge < d_nav.edge.count);
    unsigned int currentPoly = d_nav.edge.poly[agent->current_edge];
    assert(currentPoly < d_nav.poly.count);//285<135
    unsigned int pointBegin = d_nav.poly.first_point_index[currentPoly];
    unsigned int pointEnd = d_nav.poly.first_point_index[currentPoly + 1];
    unsigned int destVert = d_nav.edge.destination[agent->current_edge];
    glm::vec3 dest = glm::mix(d_nav.vertex.loc1[destVert], d_nav.vertex.loc2[destVert], 0.5);
    glm::vec3 agentLoc = glm::vec3(agent->x, agent->y, agent->z);
    dest = glm::mix(dest, closestPointonLineSegment(d_nav.vertex.loc1[destVert], d_nav.vertex.loc2[destVert], agentLoc), 0.5);
    assert(agent->x >= -1.0);
    //assert(agent->y >= -1.0);
    assert(agent->z >= -1.0);
    assert(agent->x <= 1.01);
    //assert(agent->y <= -0.99);
    assert(agent->z <= -0.454);
    //If agent is within the current edge poly
    //glm::vec2 collision_force = glm::vec2(0);
    //Calculate collision force
    if(!insideConvexPoly3D2D(pointBegin, pointEnd, &agentLoc))
    {        
        //Apply collision force
        //Currently we just take midpoint between the two edges being navigated
        //unsigned int srcVert = d_nav.edge.source[agent->current_edge];
        //glm::vec3 centerLoc = glm::mix(glm::mix(d_nav.vertex.loc1[srcVert], d_nav.vertex.loc2[srcVert], 0.5), dest, 0.01);
        //collision_force = glm::normalize(glm::vec2(centerLoc.x - agent->x, centerLoc.z - agent->z));
    }
    else
    {
        //currentPoly = d_nav.edge.poly[agent->next_edge];
        //pointBegin = d_nav.poly.first_point_index[currentPoly];
        //pointEnd = d_nav.poly.first_point_index[currentPoly + 1];
        //If agent is within next edge
        float d = distPointLine(d_nav.vertex.loc1[destVert], d_nav.vertex.loc2[destVert], agentLoc);
        if (d<NEXT_EDGE_THRESHOLD)
        {
            if (agent->next_edge == UINT_MAX)
            {
                if (!exitIsOpen(agent->exit_no))
                    agent->exit_no = getNewExitLocation(rand48);
                else
                    return 1;//Kill agent, reached exit (or gone out of bounds on final edge, close enough)
            }
            else
            {
                //Progress agent
                agent->current_edge = agent->next_edge;
                agent->next_edge = d_nav.vertex.routes[agent->exit_no][d_nav.edge.destination[agent->current_edge]];
                destVert = d_nav.edge.destination[agent->current_edge];
                dest = glm::mix(d_nav.vertex.loc1[destVert], d_nav.vertex.loc2[destVert], 0.5);
                dest = glm::mix(dest, closestPointonLineSegment(d_nav.vertex.loc1[destVert], d_nav.vertex.loc2[destVert], agentLoc), 0.01);
            }
        }
    }

    //Calculate goal force
    glm::vec2 goal_force = glm::normalize(glm::vec2(dest.x - agent->x, dest.z - agent->z));

    //collision_force *= COLLISION_WEIGHT;
	goal_force *= GOAL_WEIGHT;

    agent->desvx += goal_force.x;
    agent->desvy += goal_force.y;
    //agent->desvx += collision_force.x + goal_force.x;
    //agent->desvy += collision_force.y + goal_force.y;

    return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structre of type xmachine_memory_agent. This represents a single agent instance and can be modified directly. 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_pedestrian* agent){

	glm::vec2 agent_pos = glm::vec2(agent->x, agent->z);
	glm::vec2 agent_vel = glm::vec2(agent->vx, agent->vy);
	glm::vec2 agent_steer = glm::vec2(agent->desvx, agent->desvy);

	float current_speed = length(agent_vel)+0.025f;//(powf(length(agent_vel), 1.75f)*0.01f)+0.025f;

    //apply more steer if speed is greater
	agent_vel += current_speed*agent_steer;
	float speed = length(agent_vel);
	//limit speed
	if (speed >= agent->speed){
		agent_vel = normalize(agent_vel)*agent->speed;
		speed = agent->speed;
	}

	//update position
	agent_pos += agent_vel*TIME_SCALER;

    
	//animation
	agent->animate += (agent->animate_dir * powf(speed,2.0f)*TIME_SCALER*100.0f);
	if (agent->animate >= 1)
		agent->animate_dir = -1;
	if (agent->animate <= 0)
		agent->animate_dir = 1;

	//lod
	agent->lod = 1;

	//update
	agent->x = agent_pos.x;
	agent->z = agent_pos.y;
	agent->vx = agent_vel.x;
	agent->vy = agent_vel.y;


	//bound by wrapping
    assert(!isnan(agent->x));
    assert(!isnan(agent->z));
    agent->x = glm::clamp<float>(agent->x, -1, +1);
    agent->y = glm::clamp<float>(agent->y, -1, +1);
    agent->z = glm::clamp<float>(agent->z, -1, +1);

    //Update vertical pos (plane-line intersection)
    //https://stackoverflow.com/a/18543221/1646387
    glm::vec3 start1 = d_nav.vertex.loc1[d_nav.edge.source[agent->current_edge]];
    glm::vec3 start2 = d_nav.vertex.loc2[d_nav.edge.source[agent->current_edge]];
    glm::vec3 end = d_nav.vertex.loc2[d_nav.edge.destination[agent->current_edge]];//loc1 dest may == loc1 src?
    glm::vec3 planeNormal = normalize(glm::cross(normalize(end - start1), normalize(start2 - start1)));
    planeNormal = isnan(planeNormal.y) ? glm::vec3(0, 1, 0) : planeNormal;
    glm::vec3 u = glm::vec3(0, 1, 0);
    float dot = glm::dot(planeNormal, u);
    glm::vec3 w = normalize(glm::vec3(agent->x, agent->y, agent->z) - start1);
    float dot2 = -glm::dot(planeNormal, w) / dot;
    glm::vec3 u2 = dot2*u;
    glm::vec3 intersectPt = u2 + glm::vec3(agent->x, agent->y, agent->z);
    assert(!isnan(intersectPt.y));
    //printf("%.3f ->%.3f\n", agent->y, intersectPt.y);
    agent->y = intersectPt.y;

    return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
