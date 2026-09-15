// Compile against the pinned CaveDroneSim source; do not copy or reimplement its generator.
#include "World.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char** argv)
{
    if (argc != 2)
        return 2;

    std::uint32_t seed;
    try
    {
        const std::string argument(argv[1]);
        std::size_t used = 0;
        const auto parsed = std::stoull(argument, &used);
        if (used != argument.size() || parsed > std::numeric_limits<std::uint32_t>::max())
            return 2;
        seed = static_cast<std::uint32_t>(parsed);
    }
    catch (const std::exception&)
    {
        return 2;
    }

#ifdef _WIN32
    if (_setmode(_fileno(stdout), _O_BINARY) == -1)
        return 3;
#endif

    World world;
    world.Generate(seed);
    std::cout << "CAVEDRONE 1 " << WORLD_NX << ' ' << WORLD_NY << ' '
              << WORLD_NZ << ' ' << VOXEL_SIZE << ' ' << seed << '\n';
    std::vector<char> row(static_cast<std::size_t>(WORLD_NZ));
    for (int x = 0; x < WORLD_NX; ++x)
        for (int y = 0; y < WORLD_NY; ++y)
        {
            for (int z = 0; z < WORLD_NZ; ++z)
                row[static_cast<std::size_t>(z)] = world.IsSolid(x, y, z) ? 1 : 0;
            std::cout.write(row.data(), static_cast<std::streamsize>(row.size()));
        }
    return std::cout.good() ? 0 : 3;
}
