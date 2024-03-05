
file(GLOB_RECURSE rockchips_src **/*.cpp **/*.h)
add_library(rockchips STATIC ${rockchips_src})
target_link_directories(rockchips PUBLIC rknpu/libs/aarch64)
target_link_directories(rockchips PUBLIC rga/libs/aarch64)
target_compile_definitions(rockchips PRIVATE ENABLE_RGA)
target_link_libraries(rockchips PRIVATE ${OpenCV_LIBS} rknnrt Eigen3::Eigen)
file(GLOB rockchips_include **/include)
target_include_directories(rockchips PUBLIC ${rockchips_include})