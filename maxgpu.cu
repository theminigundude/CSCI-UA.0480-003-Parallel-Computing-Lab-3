#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

//function declaration
unsigned int getmax(unsigned int *, unsigned int);
//unsigned int getmaxSeq(unsigned int *, unsigned int);

int main(int argc, char *argv[])
{
    unsigned int size = 0;  // The size of the array
    unsigned int i;  // loop index
    unsigned int * numbers; //pointer to the array

    if(argc !=2) {
       printf("usage: maxseq num\n");
       printf("num = size of the array\n");
       exit(1);
    }

    size = atol(argv[1]);

    numbers = (unsigned int *)malloc(size * sizeof(unsigned int));
    if( !numbers ) {
       printf("Unable to allocate mem for an array of size %u\n", size);
       exit(1);
    }

    srand(time(NULL)); // setting a seed for the random number generator
    // Fill-up the array with random numbers from 0 to size-1
    for( i = 0; i < size; i++) {
      numbers[i] = rand()  % size;
    }
    printf("The maximum number in the array is: %u\n", getmax(numbers, size));

    free(numbers);
    exit(0);
}

__global__ void getmaxcu(unsigned int* num, int size, int threadCount)
{
  __shared__ int localBiggest[32];
  if (threadIdx.x==0) {
    for (int i = 0; i < 32; i++) {
      localBiggest[i] = 0;
    }
  }
  __syncthreads();

	int current =  blockIdx.x *blockDim.x + threadIdx.x;   //get current thread ID
  int localBiggestCurrent = (current - blockIdx.x *blockDim.x)/32;   //get currentID's warp number
  //if current number is bigger than the biggest number so far in the warp, replace it
  if ((num[current] > localBiggest[localBiggestCurrent]) && (current < size)) {
    localBiggest[localBiggestCurrent] = num[current];
  }
  __syncthreads();

  //using only one thread, loop through all the biggest numbers in each warp
  //and return the biggest number out of them all
  if (threadIdx.x==0) {
    int biggest = localBiggest[0];
    for (int i = 1; i < 32; i++) {
      if (biggest < localBiggest[i]) {
        biggest = localBiggest[i];
      }
    }
    //once found the biggest number in this block, put back into global array
    //num with corresponding block number
    num[blockIdx.x] = biggest;
  }

}

unsigned int getmax(unsigned int num[], unsigned int size)
{
  //get max threads per block. Since the two devices on the GPU cluster are the same,
  //I only got the property from one of the device
  int maxThreadsPerBlock, block;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  maxThreadsPerBlock = prop.maxThreadsPerBlock;
  //get numbers of blocks needed depending on size and max threads per block
  block = (size / maxThreadsPerBlock) + 1;
  if (size % maxThreadsPerBlock == 0) {
    block = size / maxThreadsPerBlock;
  }

	unsigned int* device_num;
	cudaSetDevice(1);
	cudaMalloc((void **) &device_num, size*sizeof(unsigned int));

	cudaMemcpy(device_num, num, size*sizeof(unsigned int), cudaMemcpyHostToDevice);
	getmaxcu<<<block,maxThreadsPerBlock, 32>>>(device_num, size, maxThreadsPerBlock);
	cudaMemcpy(num, device_num, size*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaFree(device_num);

  //using what we calculated, get the biggest number from each block
  int answer = num[0];
  for (int i = 1; i < block; i++) {
    if (answer < num[i]) {
      answer = num[i];
    }
  }
  return answer;
}
