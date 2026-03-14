# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-src"
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-build"
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix"
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix/tmp"
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp"
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix/src"
  "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/federico/My project (3)/ARAP_implementation/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
