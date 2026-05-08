
include(wmtk_data)
include(jse)

option(WMTK_APP_REMESH "Eigen edge-collapse remesh application" ON)
if(WMTK_APP_REMESH)
    add_subdirectory(remesh)
endif()

option(WMTK_APP_HIERARCHY "Hierarchy application" ON)
option(WMTK_APP_HARMTET "Harmonic Triangulation app" ON)
option(WMTK_APP_INTERIOR_TET_OPT "interior tet opt application" ON)

option(WMTK_APP_UNIT_TESTS "unit tests for applications" ON)

# if(WMTK_APP_HIERARCHY)
#     add_subdirectory(hieararchy)
# endif()

if(WMTK_APP_INTERIOR_TET_OPT)
add_subdirectory(interior_tet_opt)
endif()

if(WMTK_APP_HARMTET)
    add_subdirectory(harmonic_tet)
endif()

# ###############################################################################
# WMTK Application
# ###############################################################################

add_executable(wmtk_app main.cpp)
target_link_libraries(wmtk_app PUBLIC 
    wmtk::toolkit
    nlohmann_json::nlohmann_json
    CLI11::CLI11
)

wmtk_copy_dll(wmtk_app)

if(WMTK_PYBIND)
    add_subdirectory(pywmtk)
endif()