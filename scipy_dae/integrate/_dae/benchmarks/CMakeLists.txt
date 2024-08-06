cmake_minimum_required(VERSION 3.10)
project(IDAExamples)

# Find the SUNDIALS package
find_package(SUNDIALS REQUIRED)

# Set sundials library
set(SUNDIALS_LIB SUNDIALS::core SUNDIALS::ida SUNDIALS::nvecserial -lm)

# Add executables
add_executable(brenan brenan/brenan.c)
target_link_libraries(brenan PRIVATE ${SUNDIALS_LIB})

add_executable(robertson robertson/robertson.c)
target_link_libraries(robertson PRIVATE ${SUNDIALS_LIB})

add_executable(arevalo arevalo/arevalo.c)
target_link_libraries(arevalo PRIVATE ${SUNDIALS_LIB})
