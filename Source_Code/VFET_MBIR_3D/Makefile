CC = icc #C compiler
CFLAGS = -Wall -fopenmp -O3 -ansi #Compiler flags. Can change the code optimization levels.
#CFLAGS = -Wall -fopenmp -g -ansi #Compiler flags. Can change the code optimization levels.
LIBFLAGS = -lm -lfftw3
#MACROS = -DNO_COST_CALCULATE -DNO_DEBUG #Macros
#MACROS = -DNO_DEBUG #Macros
OBJS = XT_Engine.o allocate.o randlib.o tiff.o XT_AMatrix.o XT_ForwardProject.o XT_ICD_update.o XT_Init.o XT_IOMisc.o XT_Profile.o XT_VoxUpdate.o invert.o nrutil.o XT_Prior.o XT_ObjectInit.o XT_genSinogram.o XT_MagElecDen.o XT_DensityUpdate.o XT_NHICD.o
SRCS = XT_Engine.c allocate.c randlib.c tiff.c XT_AMatrix.c XT_ForwardProject.c XT_ICD_update.c XT_Init.c XT_IOMisc.c XT_Profile.c XT_VoxUpdate.c invert.c nrutil.c XT_Prior.c XT_ObjectInit.c XT_genSinogram.c XT_MagElecDen.c XT_DensityUpdate.c XT_NHICD.c
HEADERS = allocate.h tiff.h TiffWrite.h XT_AMatrix.h XT_ForwardProject.h XT_Init.h XT_Structures.h randlib.h TiffUtilities.h typeutil.h XT_Constants.h XT_ICD_update.h XT_IOMisc.h XT_Profile.h XT_VoxUpdate.h invert.h nrutil.h XT_Prior.h XT_ObjectInit.h XT_genSinogram.h XT_MagElecDen.h XT_DensityUpdate.h XT_NHICD.h
LIB = ../lib/libvfetmbir4d.a #The generated library which contains the MBIR reconstruction routines 
$(LIB): $(OBJS)
	ar rs $(LIB) $(OBJS)
$(OBJS): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(MACROS) -c $(SRCS) $(LIBFLAGS)
clean:
	rm -f *.o
	rm -f *.gch
	rm -f $(LIB)
