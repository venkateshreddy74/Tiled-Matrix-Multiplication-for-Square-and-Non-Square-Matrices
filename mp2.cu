/*
 *                         Tiled Matrix Multiplication
 *             (MP2, Fall 2014, GPU Programming/Auburn University)
 *
 *   Compile with -DTILE_WIDTH=16 (for example) to change the tile size.
 *   Compile with -DSEED=12 (for example) to seed the random number generator.
 */

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

/* Usage message displayed when invalid command line arguments are supplied */
#define USAGE \
	"MP2 generates a random (m x k) matrix M and (k x n) matrix N\n" \
	"and multiplies M by N using tiled matrix multiplication.\n" \
	"The values of m, k, and n must be >= 1.\n" \
	"\n" \
	"Usage: mp2 m k n\n"

/* Tile size -- define here if not defined using the -D compiler flag */
#ifndef TILE_WIDTH
#  define TILE_WIDTH 16
#endif

/* Seed for the random number generator -- define here if not using -D */
#ifndef SEED
#  define SEED 1
#endif

/* Maximum difference allowed between the GPU and CPU result matrices */
#define EPSILON 1e-2

/* If a CUDA call fails, display an error message and exit */
#define CUDA_CHECK(e) { \
	cudaError_t err = (e); \
	if (err != cudaSuccess) \
	{ \
		fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", \
			__FILE__, __LINE__, #e, cudaGetErrorString(err)); \
		exit(EXIT_FAILURE); \
	} \
}

/* assert() is only supported on devices of compute capability >= 2.0 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#  undef  assert
#  define assert(arg)
#endif

/* Tiled matrix multiplication kernel */
__global__  static void matMul(float *d_M, float *d_N, float *d_P,
                       int m, int k, int n)
{
	assert(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

	/*
	 *
	 *
	 * TODO: IMPLEMENT TILED MATRIX MULTIPLICATION
	 *
	 * Multiply matrix d_M by d_N, storing product in d_P.
	 * Use tiled matrix multiplication with shared memory.
	 *
	 *
	 */

//__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
//__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];


__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

int tx = threadIdx.x, ty = threadIdx.y;

int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
int col= blockIdx.x  * TILE_WIDTH + threadIdx.x;

float sum = 0;

//double numberoftiles =ceil(m/TILE_WIDTH);

if (m == k== n) {   
for (int l=0;l<m/TILE_WIDTH ; ++l) {     //iterate through tiles

	for (int j=0; j< TILE_WIDTH ; ++j) {    //iterate through elements in the tile

           sum = sum + d_M[(row*m) + ( l*TILE_WIDTH+j)] * d_N[(l*TILE_WIDTH+j)*m + col ];

         }
    
     __syncthreads();


 }

d_P[row*m +col] = sum;  

} else  {

for (int l=0;l<ceil((float) k/TILE_WIDTH) ; ++l) {     //iterate through tiles


             if (row < m  && l * TILE_WIDTH + tx < k)
		
			ds_A[ty][tx] = d_M[row *k  + l * TILE_WIDTH + tx];
	    else
			ds_A[ty][tx] = 0.0;

           if (l * TILE_WIDTH + ty < k  && col < n)

		ds_B[ty][tx] = d_N[(l * TILE_WIDTH + ty) *n  + col];
            
	 else
		ds_B[ty][tx] = 0.0;  
        
	 __syncthreads();  

        for (int j=0; j< TILE_WIDTH &&  j < k ; ++j) {    //iterate through elements in the tile

           sum = sum +  ds_A[ty][j] * ds_B[j][tx];


         }

     __syncthreads();


 }

 if (row < m  && col < n)

d_P[row*n+col] =sum;


}          


}




/* Displays one row of the given matrix */
static void printRow(int row, float *matrix, int cols)
{
	printf("[");
	if (cols >= 1) printf(" %3.3f", matrix[row*cols+0]);
	if (cols >= 2) printf(" %3.3f", matrix[row*cols+1]);
	if (cols >= 3) printf(" %3.3f", matrix[row*cols+2]);
	if (cols >= 6) printf(" ...");
	if (cols >= 5) printf(" %3.3f", matrix[row*cols+(cols-2)]);
	if (cols >= 4) printf(" %3.3f", matrix[row*cols+(cols-1)]);
	printf(" ]\n");
}

/* Displays the given matrix */
static void printMatrix(float *matrix, int rows, int cols)
{
	if (rows >= 1) printRow(0, matrix, cols);
	if (rows >= 2) printRow(1, matrix, cols);
	if (rows >= 3) printRow(2, matrix, cols);
	if (rows >= 6) printf("  ...\n");
	if (rows >= 5) printRow(rows-2, matrix, cols);
	if (rows >= 4) printRow(rows-1, matrix, cols);
}

/* Program entrypoint.  Invoke with three command line arguments: m k n */
int main(int argc, char **argv)
{
	/* Get command line arguments; save as m, k, and n */
	if (argc != 4)
	{
		fprintf(stderr, USAGE);
		fprintf(stderr, "Expected 3 arguments; received %d.\n", argc-1);
		return EXIT_FAILURE;
	}
	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	int n = atoi(argv[3]);
	if (m < 1 || k < 1 || n < 1)
	{
		fprintf(stderr, USAGE);
		fprintf(stderr, "Invalid value for m, k, or n (%d, %d, %d)\n",
			m, k, n);
		return EXIT_FAILURE;
	}
	printf("Multiplying MN = P.  M is (%d x %d); N is (%d x %d); ",
		m, k, k, n);
	printf("using (%d x %d) tiles.\n", TILE_WIDTH, TILE_WIDTH);

	/********************************************/
	/* M is (m x k), N is (k x n), P is (m x n) */
	/********************************************/

	/* Compute number of bytes needed to stores matrices M, N, and P */
	size_t bytesForM = m * k * sizeof(float);
	size_t bytesForN = k * n * sizeof(float);
	size_t bytesForP = m * n * sizeof(float);

	/* Allocate host memory for matrices */
	float *h_M, *h_N, *h_P;
	h_M = (float *)malloc(bytesForM);
	h_N = (float *)malloc(bytesForN);
	h_P = (float *)malloc(bytesForP);
	if (h_M == NULL || h_N == NULL || h_P == NULL)
	{
		fprintf(stderr, "Unable to allocate host memory\n");
		return EXIT_FAILURE;
	}

	/* Allocate device memory for matrices */
	float *d_M, *d_N, *d_P;
	CUDA_CHECK(cudaMalloc((void **)&d_M, bytesForM));
	CUDA_CHECK(cudaMalloc((void **)&d_N, bytesForN));
	CUDA_CHECK(cudaMalloc((void **)&d_P, bytesForP));

	/* Fill M and N with random numbers (on host) */
	srand(SEED);
	for (int i = 0; i < m*k; ++i)
		h_M[i] = rand()/(float)RAND_MAX*10.0;
	for (int i = 0; i < k*n; ++i)
		h_N[i] = rand()/(float)RAND_MAX*10.0;

	if (m <= 5 && k <= 5 && n <= 5)
	{
		printf("M =\n"); printMatrix(h_M, m, k);
		printf("N =\n"); printMatrix(h_N, k, n);
	}

	/* Copy M and N to device global memory */
	CUDA_CHECK(cudaMemcpy(d_M, h_M, bytesForM, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_N, h_N, bytesForN, cudaMemcpyHostToDevice));

	/* Launch the CUDA kernel */
	dim3 dimGrid((n+TILE_WIDTH-1)/TILE_WIDTH, (m+TILE_WIDTH-1)/TILE_WIDTH);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

        printf("matMul called from host");
    
	matMul<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, k, n);
	CUDA_CHECK(cudaDeviceSynchronize());

	/* Copy result matrix from device global memory back to host memory */
	CUDA_CHECK(cudaMemcpy(h_P, d_P, bytesForP, cudaMemcpyDeviceToHost));
          
          printf(" product received from host");
       
	if (m <= 5 && k <= 5 && n <= 5)
	{
		printf("P =\n"); printMatrix(h_P, m, n);
	}

	/* Verify that the result matrix is correct */
	for (int row = 0; row < m; row++)
	{
		for (int col = 0; col < n; col++)
		{
			float expected = 0.0;
			for (int i = 0; i < k; i++)
			{
				expected += h_M[row*k+i] * h_N[i*n+col];
			}

			float actual = h_P[row*n+col];

			if (fabs(expected - actual) > EPSILON)
			{
				fprintf(stderr, "d_P[%d, %d] is incorrect\n",
					row, col);
				fprintf(stderr, "    Expected: %f\n", expected);
				fprintf(stderr, "    Computed: %f\n", actual);
				return EXIT_FAILURE;
			}
		}
	}

	/* Free device global memory */
	CUDA_CHECK(cudaFree(d_M));
	CUDA_CHECK(cudaFree(d_N));
	CUDA_CHECK(cudaFree(d_P));

	/* Free host memory */
	free(h_M);
	free(h_N);
	free(h_P);

	/* Reset the device (unnecessary if not profiling, but good practice) */
	CUDA_CHECK(cudaDeviceReset());

	printf("Done\n");
	return EXIT_SUCCESS;
}
