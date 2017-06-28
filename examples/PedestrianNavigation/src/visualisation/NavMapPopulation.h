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
#ifndef _NAVMAP_POPULATION
#define _NAVMAP_POPULATION

#include "CustomVisualisation.h"

/** initNavMapPopulation
 * Initialises the navigation Map Population by loading model data and creating appropriate buffer objects and shaders
 * @param use_large_vbo determins if arrows should be instanced or displayed using a single large vbo
 */
void initNavMapPopulation(const char * modelPath);

/** renderNavMapModel
 * Renders the navigation Map Population by outputting agent data to a texture buffer object and then using vertex texture instancing 
 */
void renderNavMapPopulation();

/** Basic flat shading v/f shaders */
static const char navmap_vshader_source[] = 
{  
    "#version 120																	\n"
    "varying vec3 u_normal;									                		\n"
    "void main()																	\n"
    "{																				\n"
    "   //pass gl_Vertex to frag shader to calculate face normal                	\n"
    "	u_normal = (gl_ModelViewMatrix * gl_Vertex.xzyw).rgb;            		    \n"
    "	//apply model view proj                     								\n"
    "	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xzy,1.0);	    \n"//Swap y and z in model, because FLAME Z up
    "   //Pass gl_Color to frag shader  											\n"
    "   gl_FrontColor = gl_Color;                                           		\n"
    "}																				\n"
};
static const char navmap_fshader_source[] =
{
    "#version 120																	\n"
    "varying vec3 u_normal;   				                						\n"
    "varying vec3 o_color;              											\n"
    "void main()																	\n"
    "{																				\n"
    "   //Calculate face normal                                             		\n"
    "	vec3 N  = normalize(cross(dFdx(u_normal), dFdy(u_normal)));//Face Normal	\n"
    "	//This sets the Light source to be the camera						    	\n"
    "	vec3 L = normalize(vec3(0,0,0)-u_normal);									\n"
    "   vec3 t_color = (gl_Color.xyz==vec3(0.0f))?o_color:gl_Color.xyz;		    	\n"
    "   vec3 diffuse = t_color * max(dot(L, N), 0.0);	                            \n"
    "   gl_FragColor = vec4(diffuse.xyz,1);	                                    	\n"
    "}																				\n"
};
#endif _NAVMAP_POPULATION