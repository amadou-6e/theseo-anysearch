#include "voxelization.hpp"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <iostream>

namespace voxelization
{

    Voxelizer::Voxelizer()
    {
        openvdb::initialize();
    }

    openvdb::FloatGrid::Ptr Voxelizer::voxelizeMesh(
        const std::vector<openvdb::Vec3s> &vertices,
        const std::vector<openvdb::Vec3I> &triangles,
        const VoxelizationParams &params)
    {
        // Create transform with specified voxel size
        openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(params.voxelSize);

        // Create empty quad array since we only have triangles
        std::vector<openvdb::Vec4I> quads;

        // Convert to signed distance field
        // Note: The last two parameters are:
        // - exteriorBandWidth (distance in voxel units from the surface to the exterior narrow band)
        // - interiorBandWidth (distance in voxel units from the surface to the interior narrow band)
        openvdb::FloatGrid::Ptr grid = openvdb::tools::meshToSignedDistanceField<openvdb::FloatGrid>(
            *transform,
            vertices,
            triangles,
            quads,
            params.bandwidth, // exterior band width
            params.bandwidth  // interior band width
        );

        grid->setName("Voxelized_Mesh");
        return grid;
    }

    openvdb::FloatGrid::Ptr Voxelizer::voxelizeFile(
        const std::string &filepath,
        const VoxelizationParams &params)
    {
        std::vector<openvdb::Vec3s> vertices;
        std::vector<openvdb::Vec3I> triangles;

        // Load the mesh
        if (!loadMesh(filepath, vertices, triangles))
        {
            throw VoxelizationError("Failed to load mesh from file: " + filepath);
        }

        return voxelizeMesh(vertices, triangles, params);
    }

    bool Voxelizer::loadMesh(
        const std::string &filepath,
        std::vector<openvdb::Vec3s> &vertices,
        std::vector<openvdb::Vec3I> &triangles)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                    filepath.c_str());

        if (!warn.empty())
        {
            std::cerr << "Warning: " << warn << std::endl;
        }

        if (!err.empty())
        {
            std::cerr << "Error: " << err << std::endl;
            return false;
        }

        if (!ret)
        {
            return false;
        }

        // Convert vertices
        vertices.reserve(attrib.vertices.size() / 3);
        for (size_t i = 0; i < attrib.vertices.size(); i += 3)
        {
            vertices.emplace_back(
                attrib.vertices[i],
                attrib.vertices[i + 1],
                attrib.vertices[i + 2]);
        }

        // Convert faces to triangles
        for (const auto &shape : shapes)
        {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
            {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv == 3)
                { // Triangle
                    triangles.emplace_back(
                        shape.mesh.indices[index_offset].vertex_index,
                        shape.mesh.indices[index_offset + 1].vertex_index,
                        shape.mesh.indices[index_offset + 2].vertex_index);
                }
                index_offset += fv;
            }
        }

        return true;
    }

    void Voxelizer::saveGrid(const openvdb::GridBase::Ptr &grid, const std::string &filename) const
    {
        openvdb::io::File file(filename);
        openvdb::GridPtrVec grids;
        grids.push_back(grid);
        file.write(grids);
        file.close();
    }

    openvdb::BoolGrid::Ptr Voxelizer::convertToBinary(
        const openvdb::FloatGrid::Ptr &distanceGrid,
        float isovalue) const
    {
        auto binaryGrid = openvdb::BoolGrid::create(false);
        binaryGrid->setTransform(distanceGrid->transformPtr());

        // Convert based on inside/outside
        auto accessor = binaryGrid->getAccessor();
        for (auto iter = distanceGrid->beginValueOn(); iter; ++iter)
        {
            if (iter.getValue() < isovalue)
            {
                accessor.setValueOn(iter.getCoord(), true);
            }
        }

        return binaryGrid;
    }

} // namespace voxelization