# the dassl source files can be downloaded from https://www.ufrgs.br/dequi-app/enqlib/numeric/
cmake_minimum_required(VERSION 3.5)

project(
    "dasslc"
    VERSION "0.0.1"
    DESCRIPTION "DASSLC (Differential-Algebraic System Solver in C) is an easy-to-use and application independent C-based DAE solver package."
    HOMEPAGE_URL "http://www.ufrgs.br/dequi-app/enqlib/numeric/dasslc_v39.zip"
    LANGUAGES C)

set(dassl_sources
    src/dasslc.h
    src/dasslc.c
)
add_library(dassl ${dassl_sources})

add_executable(demo examples/demo.c)
target_link_libraries(demo dassl m)

add_executable(demo2 examples/demo2.c)
target_link_libraries(demo2 dassl m)

add_executable(demo_rt examples/demo_rt.c)
target_link_libraries(demo_rt dassl m)

add_executable(pendulum examples/pend.c)
target_link_libraries(pendulum dassl m)

# add_executable(pendulum pend.c dasslc.c dasslc.h)
# target_link_libraries(pendulum m)

# add_executable(demo demo.c dasslc.c dasslc.h)
# target_link_libraries(demo m)