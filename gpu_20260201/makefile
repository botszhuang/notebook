EXE_NAME := app.exe
NVCC := /usr/local/cuda/bin/nvcc

SRC_DIRS := .
OBJ_DIRS := .

TARGET := $(OBJ_DIRS)/$(EXE_NAME)

SRCs := $(shell find $(SRC_DIRS) -name *.c -or -name *.cpp -or -name *.cu) 
cINC_DIRS := $(shell find $(SRC_DIRS) -type d)
cINC_FLAGS := $(addprefix -I,$(cINC_DIRS))

NVCC_FLAGS := -O2 -arch=sm_75 $(cINC_FLAGS) -diag-suppress 611

all: $(TARGET)

$(TARGET): $(SRCs)
	$(NVCC) $(NVCC_FLAGS) $(OPENCV_CFLAGS) -o $@ $^ $(cINC_FLAGS)

clean:
	rm -f $(TARGET)
