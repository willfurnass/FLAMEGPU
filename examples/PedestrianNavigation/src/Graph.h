#ifndef __Graph_h__
#define __Graph_h__

#pragma warning( push )
#pragma warning( disable : 4201)
#include <glm/detail/type_vec3.hpp>
#pragma warning( pop )

struct Graph
{
    /**
    * Represents a vertex, these are shared edges of polygons (portals)
    */
    struct vertex
    {
#ifndef __CUDACC__
        vertex()
            : count(0), entryCount(0), loc1(nullptr), loc2(nullptr), routes(nullptr), first_edge_index(nullptr) { }
#endif
        unsigned int count; //@todo this might be better elsewhere, so it can be accessed via constant cache
        unsigned int entryCount;
        glm::vec3 *loc1;//Could refer these to the point array, they should be contiguous
        glm::vec3 *loc2;//They won't be contiguous if last and first vertices produce the edge
        unsigned int **routes;//2D routing table, 1st dimension is length entryCount, 2nd is lengthCount, Values are edge IDs?
                              // CSR data structure variable
        unsigned int *first_edge_index;//Len+1
    } vertex;
    /**
    * Represents an edge, these are single-direction routes between portals
    */
    struct edge
    {
#ifndef __CUDACC__
        edge()
            : count(0), source(nullptr), destination(nullptr), poly(nullptr) { }
#endif
        unsigned int count;
        unsigned int *source;
        unsigned int *destination;
        unsigned int *poly;
    } edge;
    struct poly
    {
#ifndef __CUDACC__
        poly()
            : count(0), first_point_index(nullptr) { }
#endif
        unsigned int count;
        unsigned int *first_point_index;//Len+1
    } poly;
    /**
    * Represents a spatially located polygon
    */
    struct point
    {
#ifndef __CUDACC__
        point()
            : count(0), loc(nullptr) { }
#endif
        unsigned int count;
        glm::vec3 *loc;
    } point;
    void free();
    void save(const char *path) const;
    void load(const char *path);
};

#endif //__Graph_h__