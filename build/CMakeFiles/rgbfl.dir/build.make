# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kabazoka/PC-Final

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kabazoka/PC-Final/build

# Include any dependencies generated for this target.
include CMakeFiles/rgbfl.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rgbfl.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rgbfl.dir/flags.make

CMakeFiles/rgbfl.dir/src/main.cpp.o: CMakeFiles/rgbfl.dir/flags.make
CMakeFiles/rgbfl.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kabazoka/PC-Final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rgbfl.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rgbfl.dir/src/main.cpp.o -c /home/kabazoka/PC-Final/src/main.cpp

CMakeFiles/rgbfl.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rgbfl.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kabazoka/PC-Final/src/main.cpp > CMakeFiles/rgbfl.dir/src/main.cpp.i

CMakeFiles/rgbfl.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rgbfl.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kabazoka/PC-Final/src/main.cpp -o CMakeFiles/rgbfl.dir/src/main.cpp.s

# Object files for target rgbfl
rgbfl_OBJECTS = \
"CMakeFiles/rgbfl.dir/src/main.cpp.o"

# External object files for target rgbfl
rgbfl_EXTERNAL_OBJECTS =

rgbfl: CMakeFiles/rgbfl.dir/src/main.cpp.o
rgbfl: CMakeFiles/rgbfl.dir/build.make
rgbfl: /usr/lib/x86_64-linux-gnu/libcudart_static.a
rgbfl: /usr/lib/x86_64-linux-gnu/librt.so
rgbfl: /usr/lib/x86_64-linux-gnu/libgmp.so
rgbfl: /usr/lib/x86_64-linux-gnu/libcudart_static.a
rgbfl: /usr/lib/x86_64-linux-gnu/librt.so
rgbfl: libcuda_interpolation.a
rgbfl: /usr/lib/x86_64-linux-gnu/libgmpxx.so
rgbfl: /usr/lib/x86_64-linux-gnu/libmpfr.so
rgbfl: /usr/lib/x86_64-linux-gnu/libgmp.so
rgbfl: /usr/lib/x86_64-linux-gnu/libcudart_static.a
rgbfl: /usr/lib/x86_64-linux-gnu/librt.so
rgbfl: CMakeFiles/rgbfl.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kabazoka/PC-Final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable rgbfl"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rgbfl.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rgbfl.dir/build: rgbfl

.PHONY : CMakeFiles/rgbfl.dir/build

CMakeFiles/rgbfl.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rgbfl.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rgbfl.dir/clean

CMakeFiles/rgbfl.dir/depend:
	cd /home/kabazoka/PC-Final/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kabazoka/PC-Final /home/kabazoka/PC-Final /home/kabazoka/PC-Final/build /home/kabazoka/PC-Final/build /home/kabazoka/PC-Final/build/CMakeFiles/rgbfl.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rgbfl.dir/depend

