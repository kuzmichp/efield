################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/main.cpp 

CU_SRCS += \
../src/simpleCUDA2GL.cu 

CU_DEPS += \
./src/simpleCUDA2GL.d 

OBJS += \
./src/main.o \
./src/simpleCUDA2GL.o 

CPP_DEPS += \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/samba/kuzmichp/Documents/GPU/efield/ElectricFieldVisualization" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/samba/kuzmichp/Documents/GPU/efield/ElectricFieldVisualization" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/samba/kuzmichp/Documents/GPU/efield/ElectricFieldVisualization" -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/opt/cuda/bin/nvcc -I"/opt/cuda/samples/3_Imaging" -I"/opt/cuda/samples/common/inc" -I"/home/samba/kuzmichp/Documents/GPU/efield/ElectricFieldVisualization" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


