# Copyright 2009 Matthew Cox and Earl Kinney
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

RECON_CCFILES = main.cpp bmploader.cpp nn.cpp bdp_cpu.cpp
RECON_CUFILES = nn_gpu.cu
RETAR_CCFILES = main_retarget.cpp bmploader.cpp nn.cpp bdp_cpu.cpp
RETAR_CUFILES = bdp_gpu.cu

CC_DEPS  := $(wildcard *.hpp)
CU_DEPS  := $(wildcard *.hpp)

RECON_EXECUTABLE = reconstruct
RETAR_EXECUTABLE = retarget

SRCDIR = ./
BINDIR = ./bin/
OBJDIR = ./obj/


CUDA_INSTALL_PATH ?= /opt/cuda-2.2
CUDA_SDK_INSTALL_PATH ?= /opt/cudasdk-2.2/

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := g++
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_SDK_INSTALL_PATH)/common/inc

# Warning flags
CXXWARN_FLAGS := \
        -W -Wall -Wextra -Weffc++ \
        -Wimplicit \
        -Wswitch \
        -Wformat \
        -Wchar-subscripts \
        -Wparentheses \
        -Wmultichar \
        -Wtrigraphs \
        -Wpointer-arith \
        -Wcast-align \
        -Wreturn-type \
        -Wno-unused-function \
        $(SPACE)



# Compiler-specific flags
NVCCFLAGS := --compiler-options -W,-Wall,-Wextra
CXXFLAGS  := $(CXXWARN_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

ifeq ($(dbg),1)
   COMMONFLAGS += -g
   NVCCFLAGS   += -D_DEBUG
else
   COMMONFLAGS += -O2
endif

# Libs
LIB := -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_SDK_INSTALL_PATH)/lib -lcudart -lcutil
#LIB := 



#TARGET    := $(BINDIR)/$(EXECUTABLE)
TARGET    := all

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)


#####################
# Set up object files

RECON_OBJS += $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(RECON_CCFILES)))
RECON_OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(RECON_CUFILES)))
RETAR_OBJS += $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(RETAR_CCFILES)))
RETAR_OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(RETAR_CUFILES)))


#####################
# Rules

all: reconstruct retarget

$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(CC_DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu.o : $(SRCDIR)%.cu $(CU_DEPS)
	$(NVCC) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<


reconstruct : makedirectories $(RECON_OBJS)
	$(LINK) -o $(BINDIR)/$(RECON_EXECUTABLE) $(RECON_OBJS) $(LIB)

retarget : makedirectories $(RETAR_OBJS)
	$(LINK) -o $(BINDIR)/$(RETAR_EXECUTABLE) $(RETAR_OBJS) $(LIB)



makedirectories:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

tidy:
	find . | egrep "#" | xargs rm -f
	find . | egrep "\~" | xargs rm -f

clean: tidy
	-rm -f $(RETAR_OBJS)
	-rm -f $(RECON_OBJS)
	-rm -f $(RECON_EXECUTABLE)
	-rm -f $(RETAR_EXECUTABLE)

clobber: clean
	-rm -rf $(OBJDIR)
	-rm -rf $(BINDIR)
