# Create library for voxelization
add_library(voxelization
    src/voxelization.cpp
)

target_include_directories(voxelization
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${OpenVDB_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}  # If using PCL
)

target_link_libraries(voxelization
    PUBLIC
        OpenVDB::openvdb
        TBB::tbb
        ${PCL_LIBRARIES}  # If using PCL
)

# Create main executable
add_executable(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        voxelization
)