file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS
    "src/EigenEdgeCollapse.cpp"
    "src/EigenEdgeCollapse.hpp"
    "src/EigenEdgeCollapseInternal.hpp"
    "src/FemAssembly.cpp"
    "src/CostFunctions.cpp"
    "src/Visualization.cpp"
    "src/VisUtils.cpp"
    "src/VisUtils.hpp"
    # stb_image_write impl is provided transitively by polyscope's bundled
    # stb library (linked via wmtk::toolkit → polyscope → stb).
)

add_library(remesh_lib ${LIB_SOURCES})
add_library(wmtk::remesh ALIAS remesh_lib)
target_link_libraries(remesh_lib PUBLIC
    wmtk::toolkit
    igl::core
)
target_include_directories(remesh_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# SuiteSparse / CHOLMOD (for Eigen::CholmodSupernodalLLT)
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_ROOT "${LIBOMP_PREFIX}" CACHE PATH "" FORCE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
endif()

find_package(OpenMP REQUIRED)
target_link_libraries(remesh_lib PUBLIC OpenMP::OpenMP_CXX)


execute_process(
    COMMAND brew --prefix
    OUTPUT_VARIABLE HOMEBREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# SuiteSparse 同梱の FindCBLAS.cmake などを見つけられるように
list(APPEND CMAKE_MODULE_PATH "${HOMEBREW_PREFIX}/lib/cmake/SuiteSparse")
list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")

# find_package(SuiteSparse REQUIRED)
find_package(CHOLMOD CONFIG REQUIRED)

target_link_libraries(remesh_lib PUBLIC
    SuiteSparse::CHOLMOD
    # SuiteSparse::SPQR
)
# target_include_directories(remesh_lib PUBLIC ${SuiteSparse_INCLUDE_DIRS})
# target_link_libraries(remesh_lib PUBLIC ${SuiteSparse_LIBRARIES})

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)






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
    ${ACCELERATE_FRAMEWORK}
)
wmtk_copy_dll(wmtk_remesh)
