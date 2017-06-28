#include "Graph.h"
#include <fstream>

void Graph::free()
{
    if (this->vertex.loc1)
        std::free(this->vertex.loc1);
    if (this->vertex.loc2)
        std::free(this->vertex.loc2);
    if (this->vertex.routes)
        for (unsigned int i = 0; i<this->vertex.entryCount; ++i)
            std::free(this->vertex.routes[i]);
    std::free(this->vertex.routes);
    if (this->vertex.first_edge_index)
        std::free(this->vertex.first_edge_index);
    this->vertex.count = 0;
    this->vertex.entryCount = 0;
    if (this->edge.source)
        std::free(this->edge.source);
    if (this->edge.destination)
        std::free(this->edge.destination);
    if (this->edge.poly)
        std::free(this->edge.poly);
    this->edge.count = 0;
    if (this->poly.first_point_index)
        std::free(this->poly.first_point_index);
    this->poly.count = 0;
    if (this->point.loc)
        std::free(this->point.loc);
    this->point.count = 0;
}

const unsigned int TWELVE = 12;
void Graph::save(const char *path) const
{
    std::ofstream outfile(path, std::ofstream::binary);

    outfile.write((char*)&this->vertex.count, sizeof(unsigned int));
    outfile.write((char*)&this->vertex.entryCount, sizeof(unsigned int));
    outfile.write((char*)this->vertex.loc1, sizeof(glm::vec3)*this->vertex.count);
    outfile.write((char*)this->vertex.loc2, sizeof(glm::vec3)*this->vertex.count);
    for (unsigned int i = 0; i<this->vertex.entryCount; ++i)
        outfile.write((char*)this->vertex.routes[i], sizeof(unsigned int)*this->vertex.count);
    outfile.write((char*)this->vertex.first_edge_index, sizeof(unsigned int)*(this->vertex.count + 1));

    outfile.write((char*)&this->edge.count, sizeof(unsigned int));
    outfile.write((char*)this->edge.source, sizeof(unsigned int)*this->edge.count);
    outfile.write((char*)this->edge.destination, sizeof(unsigned int)*this->edge.count);
    outfile.write((char*)this->edge.poly, sizeof(unsigned int)*this->edge.count);

    outfile.write((char*)&this->poly.count, sizeof(unsigned int));
    outfile.write((char*)this->poly.first_point_index, sizeof(unsigned int)*(this->poly.count + 1));

    outfile.write((char*)&this->point.count, sizeof(unsigned int));
    outfile.write((char*)this->point.loc, sizeof(glm::vec3)*this->point.count);

    outfile.write((char*)TWELVE, sizeof(TWELVE));//Used to ensure file is right length

    outfile.close();
}
void Graph::load(const char *path)
{
    this->free();//This might be a bad idea if the struct is shared
    std::ifstream infile(path, std::ifstream::binary);
    if (!infile.is_open())
    {
        fprintf(stderr, "File '%s' does not exist.\n", path);
        exit(EXIT_FAILURE);
    }
    infile.read((char*)&this->vertex.count, sizeof(unsigned int));
    infile.read((char*)&this->vertex.entryCount, sizeof(unsigned int));

    this->vertex.loc1 = (glm::vec3*)malloc(sizeof(glm::vec3) * this->vertex.count);
    this->vertex.loc2 = (glm::vec3*)malloc(sizeof(glm::vec3) * this->vertex.count);
    this->vertex.routes = (unsigned int**)malloc(sizeof(unsigned int*) * this->vertex.entryCount);
    for (unsigned int entry = 0; entry < this->vertex.entryCount; ++entry)
    {
        this->vertex.routes[entry] = (unsigned int *)malloc(this->vertex.count * sizeof(unsigned int));
    }
    this->vertex.first_edge_index = (unsigned int*)malloc((this->vertex.count + 1) * sizeof(unsigned int));

    infile.read((char*)this->vertex.loc1, sizeof(glm::vec3) * this->vertex.count);
    infile.read((char*)this->vertex.loc2, sizeof(glm::vec3) * this->vertex.count);
    for (unsigned int i = 0; i<this->vertex.entryCount; ++i)
        infile.read((char*)this->vertex.routes[i], sizeof(unsigned int) * this->vertex.count);
    infile.read((char*)this->vertex.first_edge_index, sizeof(unsigned int) * (this->vertex.count + 1));

    infile.read((char*)&this->edge.count, sizeof(unsigned int));

    this->edge.source = (unsigned int*)malloc(this->edge.count * sizeof(unsigned int));
    this->edge.destination = (unsigned int*)malloc(this->edge.count * sizeof(unsigned int));
    this->edge.poly = (unsigned int*)malloc(this->edge.count * sizeof(unsigned int));

    infile.read((char*)this->edge.source, sizeof(unsigned int) * this->edge.count);
    infile.read((char*)this->edge.destination, sizeof(unsigned int) * this->edge.count);
    infile.read((char*)this->edge.poly, sizeof(unsigned int) * this->edge.count);

    infile.read((char*)&this->poly.count, sizeof(unsigned int));

    this->poly.first_point_index = (unsigned int *)malloc((this->poly.count + 1) * sizeof(unsigned int));

    infile.read((char*)this->poly.first_point_index, sizeof(unsigned int)*(this->poly.count + 1));

    infile.read((char*)&this->point.count, sizeof(unsigned int));

    this->point.loc = (glm::vec3 *)malloc(this->point.count * sizeof(glm::vec3));

    infile.read((char*)this->point.loc, sizeof(glm::vec3)*this->point.count);

    unsigned int verification;
    infile.read((char*)&verification, sizeof(TWELVE));
    assert(verification == TWELVE);

    infile.close();
}