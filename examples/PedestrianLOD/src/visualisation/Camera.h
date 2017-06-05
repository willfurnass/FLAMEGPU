#ifndef __Camera_h__
#define __Camera_h__
#include <cstdio>

#include <GL/glew.h>
#include <GL/glut.h>
#ifdef _DEBUG //VS standard debug flag

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

#define GL_CALL( err ) err ;HandleGLError(__FILE__, __LINE__)
#define GL_CHECK() (HandleGLError(__FILE__, __LINE__))

#else //ifdef _DEBUG
//Remove the checks when running release mode.
#define GL_CALL( err ) err
#define GL_CHECK() 

#endif //ifdef  _DEBUG

#include <glm/glm.hpp>

class Camera
{
public:
    Camera();
    Camera(glm::vec3 eye); 
    Camera(glm::vec3 eye, glm::vec3 target);
    ~Camera();

    void turn(float thetaInc, float phiInc);
    void move(float distance);
    void strafe(float distance);
    void ascend(float distance);
    void roll(float distance);
    void setStabilise(bool stabilise);
    glm::mat4 view() const; 
    glm::mat4 skyboxView() const;
    glm::vec3 getEye() const;
    glm::vec3 getLook() const;
    glm::vec3 getUp() const;
    glm::vec3 getPureUp() const;
    glm::vec3 getRight() const;
    const glm::mat4 *Camera::getViewMatPtr() const;
    const glm::mat4 *Camera::getSkyboxViewMatPtr() const;
private:
    void updateViews();
    //ModelView matrix
    glm::mat4 viewMat;
    //Model view matrix without camera position taken into consideration
    glm::mat4 skyboxViewMat;
    //Up vector used for stabilisation, only rotated when roll is called
    glm::vec3 pureUp;
    //Eyelocation
    glm::vec3 eye;
    //3 perpendicular vectors which represent the cameras direction and orientation
    glm::vec3 look;
    glm::vec3 right;
    glm::vec3 up;
    bool stabilise;
};

#endif //ifndef __Camera_h__