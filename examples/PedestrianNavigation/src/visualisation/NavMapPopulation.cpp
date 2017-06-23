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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <GL/glew.h>
#include <GL/glut.h>
#include "NavMapPopulation.h"
#include "OBJModel.h"
#include "BufferObjects.h"
#include <glm/gtc/type_ptr.hpp>

//navigation map width
int nm_width;

//navmap instances
GLuint nm_instances_tbo;
GLuint nm_instances_tex;

//model primative counts
int arrow_v_count;
int arrow_f_count;
//model primative data
glm::vec3* arrow_vertices;
glm::vec3* arrow_normals;
glm::ivec3* arrow_faces;
//model buffer obejcts
GLuint arrow_verts_vbo;
GLuint arrow_elems_vbo;

//Shader and shader attribute pointers
GLuint nm_vertexShader, nm_fragShader;
GLuint nm_shaderProgram;
GLuint nmvs_instance_map;
GLuint nmvs_instance_index;
GLuint nmvs_NM_WIDTH;
GLuint nmvs_ENV_MAX;
GLuint nmvs_ENV_WIDTH;

//external prototypes imported from FLAME GPU

//PRIVATE PROTOTYPES
/** createNavMapBufferObjects
 * Creates all Buffer Objects for instancing and model data
 */
void createNavMapBufferObjects();
/** initNavMapShader
 * Initialises the Navigation Map Shader and shader attributes
 */
void initNavMapShader();

void initNavMapPopulation()
{
	////load cone model
	allocateObjModel("C:\\Users\\rob\\recastgit\\RecastDemo\\Bin\\Meshes\\rotate_underground.obj", &arrow_v_count, &arrow_f_count, &arrow_vertices, &arrow_normals, &arrow_faces);
    loadObjFromFile("C:\\Users\\rob\\recastgit\\RecastDemo\\Bin\\Meshes\\rotate_underground.obj", arrow_v_count, arrow_f_count, arrow_vertices, arrow_normals, arrow_faces);

	//scaleObj(scale, arrow_v_count, arrow_vertices);		 

	createNavMapBufferObjects();
	initNavMapShader();
}
void rescaleNavMapPopulation(float scaleFactor, glm::vec3 offset)
{
    //scale obj
    scaleObjwithOffset(scaleFactor, offset, arrow_v_count, arrow_vertices);
    //update vbos
    glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo);
    glBufferData(GL_ARRAY_BUFFER, arrow_v_count*sizeof(glm::vec3), arrow_vertices, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    for (unsigned int i = 0; i < arrow_f_count; ++i)
    {
        assert(arrow_faces[i].x < arrow_v_count);
        assert(arrow_faces[i].y < arrow_v_count);
        assert(arrow_faces[i].z < arrow_v_count);
    }
}
#include "Graph.h"
extern Graph h_nav;
inline static void HandleGLError(const char *file, int line) {
    GLuint error = glGetError();
    if (error != GL_NO_ERROR)
    {
        printf("%s(%i) GL Error Occurred;\n%s\n", file, line, gluErrorString(error));
#if EXIT_ON_ERROR
        getchar();
        exit(1);
#endif
    }
}

#define GL_CALL( err ) err //;HandleGLError(__FILE__, __LINE__)
#define GL_CHECK() (HandleGLError(__FILE__, __LINE__))
bool renderModel = true;
void toggleRenderModel()
{
    renderModel = !renderModel;
}
void renderNavMapPopulation()
{
    //Render the 3D model
    if (renderModel)
    {
        GL_CALL(glUseProgram(nm_shaderProgram));
        GL_CALL(glColor3f(1, 1, 1));//White
        GL_CALL(glEnableClientState(GL_VERTEX_ARRAY));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo));
        GL_CALL(glVertexPointer(3, GL_FLOAT, 0, 0));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo));
        GL_CALL(glDrawElements(GL_TRIANGLES, arrow_f_count * 3, GL_UNSIGNED_INT, 0));
        GL_CALL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        GL_CALL(glDisableClientState(GL_VERTEX_ARRAY));
        GL_CALL(glBindBuffer(GL_ARRAY_BUFFER, 0));
        GL_CALL(glUseProgram(0));
    }
    ////Render the mesh
    for (unsigned int poly = 0; poly < h_nav.poly.count; ++poly)
    {//For each poly
        GL_CALL(glBegin(GL_LINE_LOOP));
        for (unsigned int edge = h_nav.poly.first_point_index[poly]; edge < h_nav.poly.first_point_index[poly+1]; ++edge)
        {//For each vertex
            GL_CALL(glColor3f(0, 0, 0));//Black
            GL_CALL(glVertex3f(h_nav.point.loc[edge].x, h_nav.point.loc[edge].z, h_nav.point.loc[edge].y));
        }
        GL_CALL(glEnd());
    }
}


void createNavMapBufferObjects()
{
	//create VBOs
	createVBO(&arrow_verts_vbo, GL_ARRAY_BUFFER, arrow_v_count*sizeof(glm::vec3));
	createVBO(&arrow_elems_vbo, GL_ELEMENT_ARRAY_BUFFER, arrow_f_count*sizeof(glm::ivec3));

	//bind VBOs
	glBindBuffer(GL_ARRAY_BUFFER, arrow_verts_vbo);
	glBufferData(GL_ARRAY_BUFFER, arrow_v_count*sizeof(glm::vec3), arrow_vertices, GL_DYNAMIC_DRAW);
	glBindBuffer( GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_elems_vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, arrow_f_count*sizeof(glm::ivec3), arrow_faces, GL_DYNAMIC_DRAW);
	glBindBuffer( GL_ARRAY_BUFFER, 0);
}

void initNavMapShader()
{
    const char* v = navmap_vshader_source;
    const char* f = navmap_fshader_source;
	int status;

	//vertex shader
	nm_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(nm_vertexShader, 1, &v, 0);
    glCompileShader(nm_vertexShader);

    nm_fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(nm_fragShader, 1, &f, 0);
    glCompileShader(nm_fragShader);

	//program
    nm_shaderProgram = glCreateProgram();
    glAttachShader(nm_shaderProgram, nm_vertexShader);
    glAttachShader(nm_shaderProgram, nm_fragShader);
    glLinkProgram(nm_shaderProgram);

	// check for errors
	glGetShaderiv(nm_vertexShader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE){
		char data[1024];
		int len;
		printf("ERROR: Shader Compilation Error\n");
		glGetShaderInfoLog(nm_vertexShader, 1024, &len, data); 
		printf("%s", data);
    }
    glGetShaderiv(nm_fragShader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE){
        char data[1024];
        int len;
        printf("ERROR: Shader Compilation Error\n");
        glGetShaderInfoLog(nm_fragShader, 1024, &len, data);
        printf("%s", data);
    }
	glGetProgramiv(nm_shaderProgram, GL_LINK_STATUS, &status);
	if (status == GL_FALSE){
		printf("ERROR: Shader Program Link Error\n");
	}
}
