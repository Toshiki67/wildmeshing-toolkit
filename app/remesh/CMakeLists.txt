file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "src/EigenEdgeCollapse.cpp"
    "src/EigenEdgeCollapse.hpp"
    "src/VisUtils.cpp"
    "src/VisUtils.hpp"
    "src/stb_impl.cpp"
)

add_library(remesh_lib ${LIB_SOURCES})
add_library(wmtk::remesh ALIAS remesh_lib)
target_link_libraries(remesh_lib PUBLIC
    wmtk::toolkit
    igl::core
)
target_include_directories(remesh_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# stb_image_write (via polyscope's bundled copy)
find_path(STB_INCLUDE_DIR
    NAMES stb_image_write.h
    HINTS ${CMAKE_BINARY_DIR}/_deps/polyscope-src/deps/stb
    REQUIRED
)
target_include_directories(remesh_lib PRIVATE ${STB_INCLUDE_DIR})

add_executable(wmtk_remesh main.cpp)
target_link_libraries(wmtk_remesh PUBLIC
    remesh_lib
    CLI11::CLI11
)
wmtk_copy_dll(wmtk_remesh)
