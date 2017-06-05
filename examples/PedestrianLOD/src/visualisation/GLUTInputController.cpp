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
#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cmath>
#include "CustomVisualisation.h"
#include "GLUTInputController.h"
#include "MenuDisplay.h"
#include "GlobalsController.h"
#include "MenuDisplay.h"
#include "Camera.h"
#include <cstring>
#define DELTA_THETA_PHI 0.01f
#define MOUSE_SPEED 0.001f
#define SHIFT_MULTIPLIER 5.0f

#define MOUSE_SPEED_FPS 0.05f
#define DELTA_MOVE 0.1f
#define DELTA_STRAFE 0.1f
#define DELTA_ASCEND 0.1f
#define DELTA_ROLL 0.01f
//viewpoint vectors and eye distance
glm::vec3 eye;

float theta;
float phi;
float cos_theta;
float sin_theta;
float cos_phi;
float sin_phi;

int mouse_old_x, mouse_old_y;

int zoom_key = 0;

#define TRANSLATION_SCALE 0.005f
#define ROTATION_SCALE 0.01f
#define ZOOM_SCALE 0.01f

#define MAX_ZOOM 0.01f
#define MIN_PHI 0.0f

#define PI 3.14
#define rad(x) (PI / 180) * x

Camera *cam;

char keys[1024];
void initInputConroller()
{
	float eye_distance = ENV_MAX*1.75f;
	cam = new Camera(glm::vec3(1, 1, eye_distance), glm::vec3(0));
	memset(keys, 0, sizeof(keys));
}

bool mouseLocked = false;
bool handleMouseMove = true;
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		mouseLocked = !mouseLocked;
		if (mouseLocked)
		{
			glutSetCursor(GLUT_CURSOR_NONE);
			handleMouseMove = true;
		}
		else
			glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}
void mouseMove(int x, int y)
{
	if (mouseLocked&&handleMouseMove)
	{
		cam->turn((x - (800 / 2)) * MOUSE_SPEED, (y - (600 / 2)) * MOUSE_SPEED);
		glutWarpPointer(800 / 2, 600 / 2);
	}
	handleMouseMove = !handleMouseMove;
}
void keyboard( unsigned char key, int x, int y)
{
	keys[1023] = tolower(key) != key;
	keys[tolower(key)] = 1;
	keys[1022] = (glutGetModifiers()&GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL;

	switch (key) {
		case('f'):
		{
			toggleFullScreenMode();
			break;
		}
		case('i'):
		{
			setMenuDisplayOnOff(0);
			toggleInformationDisplayOnOff();
			break;
		}
		case('m'):
		{
			setInformationDisplayOnOff(0);
			toggleMenuDisplayOnOff();
			break;
		}
		case('z'):
		{
			zoom_key = !zoom_key;
			break;
		}
		default:
		{
			break;
		}
    }
}
void keyboardUp(unsigned char key, int x, int y)
{
	keys[1023] = tolower(key) != key;
	keys[1022] = (glutGetModifiers()&GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL;
	keys[tolower(key)] = 0;
}
extern float frame_time;
void handleKeyBuffer()
{
	float speed = 1.0f;
	float turboMultiplier = keys[1023] ? SHIFT_MULTIPLIER*speed : speed;
	turboMultiplier /= (1000.0f / frame_time);//Adjust speed for variable fps
	if (keys['w'])
		cam->move(DELTA_MOVE*turboMultiplier);
	if (keys['a'])
		cam->strafe(-DELTA_STRAFE*turboMultiplier);
	if (keys['s'])
		cam->move(-DELTA_MOVE*turboMultiplier);
	if (keys['d'])
		cam->strafe(DELTA_STRAFE*turboMultiplier);
	if (keys['q'])
		cam->roll(-DELTA_ROLL);
	if (keys['e'])
		cam->roll(DELTA_ROLL);
	if (keys[' '])
		cam->ascend(DELTA_ASCEND*turboMultiplier);
}
void specialKeyboard(int key, int x, int y)
{
	keys[1023] = tolower(key) == key;
	if (menuDisplayed())
	{
		switch(key) {
			case(GLUT_KEY_DOWN):
			{
				handleDownKey();
				break;
			}
			case(GLUT_KEY_UP):
			{
				handleUpKey();
				break;
			}
			case(GLUT_KEY_LEFT):
			{
				handleLeftKey();
				break;
			}
			case(GLUT_KEY_RIGHT):
			{
				handleRightKey();
				break;
			}
			default:
			{
				break;
			}
		}
	}
}


void setMatrices()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glm::mat4 mv = cam->view();
	glLoadMatrixf((float*)&mv);
	eye = cam->getEye();
}