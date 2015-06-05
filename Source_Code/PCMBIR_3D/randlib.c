/* ============================================================================
 * Copyright (c) 2013 Charles A. Bouman (Purdue University)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright notice, this
 * list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * Neither the name of Charles A. Bouman, Purdue
 * University, nor the names of its contributors may be used
 * to endorse or promote products derived from this software without specific
 * prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* random3(): modified on 10/23/89 from random2() to generate positive ints*/
/* readsead and writeseed: modified on 4/9/91 to read and write from /tmp */
/* modified on 12/9/93 to be ANSI compliant. */

#include "randlib.h"

void exit();

#define MAXPRIME  2147483647       /*  MAXPRIME = (2^31)-1     */
#define PI        3.14159265358979323846

/* PORTABILITY 1:  The functions in this file assume that a long is 32 bits
      and a short is 16 bits.  These conventions are machine dependent and
      must therefore be verified before using.                     */

static long int tmp;
static uint32_t   sd[2];   /*  tmp: 31 bit seed in GF( (2^31)-1 )   */
                                    /*  sd[0]: high order 15 bits of tmp     */
                                    /*  sd[1]: low order 16 bits of tmp      */
                 /* NOTE: high order 16 bits of sd[0] and sd[1] are 0        */

/* Generates 20 random #'s starting from a seed of 1  */
/*
main()
{
 int32_t    i;
 double  r,random2();

 srandom2(1);
 for(i=0;i<20;i++) {
    r=random2();
    printf("%.8f\n",r);
 }
}
*/
/*  TABLE:  The following are the first 20 random #'s which should be generated
         starting from a seed of 1:

16807
282475249
1622650073
984943658
1144108930
470211272
101027544
1457850878
1458777923
2007237709
823564440
1115438165
1784484492
74243042
114807987
1137522503
1441282327
16531729
823378840
143542612
                                          */

/*  Test for verifying the cycle length of the random # generator  */
/*   NOTE:  to speed up this test, comment out the return statement
                               in random2()                        */
/*
main()
{
 double random2();
 long i;
 
 srandom2(1);
 tmp=0;
 while (tmp!=1) {
   for(i=0;i<(256*256*256);i++) {
       random2();
       if (tmp == 1) break;
   }
   printf("*\n");
   if (tmp == 0) break;
   writeseed();
 }
 printf("\n%d\n",i);
 writeseed();
}
*/

double random2()
/* Uniform random number generator on (0,1] */
/*  Algorithm:  newseed = (16807 * oldseed) MOD [(2^31) - 1]  ;
                returned value = newseed / ( (2^31)-1 )  ;
      newseed is stored in tmp and sd[0] and sd[1] are updated;
      Since 16807 is a primitive element of GF[(2^31)-1], repeated calls
      to random2() should generate all positive integers <= MAXPRIME
      before repeating any one value.
    Tested: Feb. 16, 1988;  verified the length of cycle of integers 
                             generated by repeated calls to random2()  */
{
 *(sd+1) *= 16807;
 *sd *= 16807;
 tmp=((*sd)>>15)+(((*sd)&0x7fff)<<16);
 tmp += (*(sd+1));
 if ( tmp & 0x80000000 ) {
   tmp++;
   tmp &= 0x7fffffff;
 }
 *sd=tmp>>16;
 *(sd+1)=tmp & 0xffff;
 return(((double)tmp)/MAXPRIME);   
}

int32_t random3()
/* random3(): modified on 10/23/89 from random2() to generate positive ints*/
/* Uniform random number generator on (0,1] */
/*  Algorithm:  newseed = (16807 * oldseed) MOD [(2^31) - 1]  ;
                returned value = newseed / ( (2^31)-1 )  ;
      newseed is stored in tmp and sd[0] and sd[1] are updated;
      Since 16807 is a primitive element of GF[(2^31)-1], repeated calls
      to random2() should generate all positive integers <= MAXPRIME
      before repeating any one value.
    Tested: Feb. 16, 1988;  verified the length of cycle of integers 
                             generated by repeated calls to random2()  */
{
 *(sd+1) *= 16807;
 *sd *= 16807;
 tmp=((*sd)>>15)+(((*sd)&0x7fff)<<16);
 tmp += (*(sd+1));
 if ( tmp & 0x80000000 ) {
   tmp++;
   tmp &= 0x7fffffff;
 }
 *sd=tmp>>16;
 *(sd+1)=tmp & 0xffff;
 return((int)tmp);   
}

void srandom2(uint32_t num)
/* Set a new seed for random # generator  */
{
 tmp=num;
 *sd=tmp>>16;
 *(sd+1)=tmp & 0xffff;
}

void readseed()
/*  Reads random # generator seed from file: /tmp/randomseed */
{
 FILE	*fp1;
 void	writeseed();

   if((fp1 = fopen("/tmp/randomseed","r"))==NULL ) {
     fprintf(stderr,"readseed: creating file /tmp/randomseed\n");
     tmp=143542612;
     writeseed();
     srandom2(tmp);
   } else {
     fscanf(fp1,"%ld",&tmp);
     srandom2(tmp);
     fclose(fp1);
   }
}

void writeseed()
/*  Writes random # generator seed from file: /tmp/randomseed */
{
 FILE  *fp1;

   if((fp1 = fopen("/tmp/randomseed","w"))==NULL ) {
     fprintf(stderr,"writeseed: can't open file /tmp/randomseed\n");
     exit(1);
   } else {
     fprintf(fp1,"%ld",tmp);
     fclose(fp1);
   }
}

double normal()
/*  Generates normal random numbers: N(0,1)  */
{
 static int32_t   even = 1;  /*   if  even = 0:  return b              */
			 /*       even = 1:  compute 2 new values  */
 static double   b;      /*   temporary storage                    */
 double a,r,theta,random2();

 if((even=!even)) {
    return(b);
 } else {
    theta=2*PI*random2();
    r=sqrt(-2*log(random2()));
    a=r*cos(theta);
    b=r*sin(theta);
    return(a);
 }
}

double dexprand()
/*  Generates a double exponentially distributed random variable
      with mean 0 and variance 2.                                 */
{
 double  a,random2();

 a= -log(random2());
 if( random2()>0.5 )   a=(-a);
 return(a);
}
    


