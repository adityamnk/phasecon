CC = mpicc #MPI compiler
CFLAGS = -Wall -fopenmp -O3 -ansi #Compiler flags. Can change the code optimization levels.
LIBFLAGS = -lm -lfftw3
#MACROS = -DNO_COST_CALCULATE -DNO_DEBUG #Macros
#MACROS = -DNO_DEBUG #Macros
OBJS = XT_Engine.o allocate.o randlib.o tiff.o XT_AMatrix.o XT_ForwardProject.o XT_ICD_update.o XT_Init.o XT_IOMisc.o XT_MPI.o XT_MPIIO.o XT_NHICD.o XT_Profile.o XT_VoxUpdate.o invert.o nrutil.o XT_OffsetError.o XT_genSinogram.o XT_Prior.o XT_Search.o XT_PhaseRet.o XT_CmplxArith.o XT_CmplxProjEst.o XT_FresnelTran.o XT_Paganin.o XT_ObjectInit.o
SRCS = XT_Engine.c allocate.c randlib.c tiff.c XT_AMatrix.c XT_ForwardProject.c XT_ICD_update.c XT_Init.c XT_IOMisc.c XT_MPI.c XT_MPIIO.c XT_NHICD.c XT_Profile.c XT_VoxUpdate.c invert.c nrutil.c XT_OffsetError.c XT_genSinogram.c XT_Prior.c XT_Search.c XT_PhaseRet.c XT_CmplxArith.c XT_CmplxProjEst.c XT_FresnelTran.c XT_Paganin.c XT_ObjectInit.c
HEADERS = allocate.h tiff.h TiffWrite.h XT_AMatrix.h XT_ForwardProject.h XT_Init.h XT_MPI.h XT_NHICD.h XT_Structures.h randlib.h TiffUtilities.h typeutil.h XT_Constants.h XT_ICD_update.h XT_IOMisc.h XT_MPIIO.h XT_Profile.h XT_VoxUpdate.h invert.h nrutil.h XT_OffsetError.h XT_genSinogram.h XT_Prior.h XT_Search.h XT_PhaseRet.h XT_CmplxArith.h XT_CmplxProjEst.h XT_FresnelTran.h XT_Paganin.h XT_ObjectInit.h
LIB = ../lib/libpcmbir4d.a #The generated library which contains the MBIR reconstruction routines 
$(LIB): $(OBJS)
	ar rs $(LIB) $(OBJS)
$(OBJS): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(MACROS) -c $(SRCS) $(LIBFLAGS)
clean:
	rm -f *.o
	rm -f *.gch
	rm -f $(LIB)
