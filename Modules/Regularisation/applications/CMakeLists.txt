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

include_directories("../../Common/include")
link_libraries(Common)
include_directories("../../Geometry/include")
link_libraries(Geometry)
include_directories("../../Image/include")
link_libraries(Image)
include_directories("../../Regularisation/include")
link_libraries(Regularisation)

macro(irtk_add_application _name)
  add_executable(${_name} ${ARGN})
  set_target_properties(${_name} PROPERTIES
                        OUTPUT_NAME irtk-${_name})
  install(TARGETS ${_name}
          DESTINATION ${IRTK_INSTALL_BINDIR})
endmacro()

irtk_add_application(denseCRFRegularisation common.cpp denseCRFRegularisation.cc)


