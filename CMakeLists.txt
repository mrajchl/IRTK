# The Image Registration Toolkit (IRTK)
#
# Copyright 2008-2015 Imperial College London
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project(IRTK)

cmake_minimum_required(VERSION 2.8)

option(BUILD_APPLICATIONS "Build applications." ON)
option(BUILD_DOCUMENTATION "Build documentation." OFF)
#option(BUILD_PACKAGES "Build packages." OFF)
#option(BUILD_SHARED_LIBS "Build libraries dynamically (ON) or statically (OFF)" OFF)
#option(BUILD_TESTING "Build test suite." OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")

include(IRTKCheckCompiler)
include(IRTKBuildDeps)
include(IRTKInstallDirs)
include(IRTKVersion)

add_subdirectory(Modules)

if(BUILD_APPLICATIONS)
  add_subdirectory(Applications)
endif()

if(BUILD_DOCUMENTATION)
  add_subdirectory(Documentation)
endif()

if(BUILD_PACKAGES)
  add_subdirectory(Packages)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
