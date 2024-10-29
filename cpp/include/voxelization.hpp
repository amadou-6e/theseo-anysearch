#pragma once

#include <openvdb/openvdb.h>
#include <openvdb/tools/MeshToVolume.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace voxelization
{

    struct VoxelizationParams
    {
        double voxelSize = 0.1; // Size of each voxel
        double bandwidth = 3.0; // Width of narrow band for SDF
    };

    class VoxelizationError : public std::runtime_error
    {
    public:
        explicit VoxelizationError(const std::string &msg)
            : std::runtime_error(msg) {}
    };

    class Voxelizer
    {
    public:
        Voxelizer();

        openvdb::FloatGrid::Ptr voxelizeFile(
            const std::string &filepath,
            const VoxelizationParams &params = VoxelizationParams());

        openvdb::FloatGrid::Ptr voxelizeMesh(
            const std::vector<openvdb::Vec3s> &vertices,
            const std::vector<openvdb::Vec3I> &triangles,
            const VoxelizationParams &params = VoxelizationParams());

        void saveGrid(
            const openvdb::GridBase::Ptr &grid,
            const std::string &filename) const;

        openvdb::BoolGrid::Ptr convertToBinary(
            const openvdb::FloatGrid::Ptr &distanceGrid,
            float isovalue = 0.0) const;

    private:
        bool loadMesh(
            const std::string &filepath,
            std::vector<openvdb::Vec3s> &vertices,
            std::vector<openvdb::Vec3I> &triangles);
    };

} // namespace voxelization