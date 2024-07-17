#include <stdio.h>
#include <stdlib.h>

#include <time.h> 

#include <cuda/std/complex>
#include <cufft.h>

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

//Macro for checking cuda errors following a cuda launch or api call
#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
   if (CUFFT_SUCCESS != err) {
      fprintf(stderr, "CUFFT error in file '%s', line %d\n \nerror %d: %s\nterminating!\n", __FILE__, __LINE__, err,
                           _cudaGetErrorEnum(err));
             cudaDeviceReset(); exit(0); \
   }
}

/*---------------------------------------------------------------------------------*/
// Definitions

# define M_PI 3.14159265358979323846
# define I cuda::std::complex<float>(0, 1)

typedef cuda::std::complex<float> complex;

/*---------------------------------------------------------------------------------*/
// Variables

int Nx, Ny, Nz, nproj = 0;
float dx, dy, dz = 0;

int BLOCKS, THREADS_PER_BLOCK = 0;

complex *sinogram = nullptr;

/*---------------------------------------------------------------------------------*/
// Function declerations

complex *load_sinogram(const char *config_file, const char *data_file);
//complex *odt_reconstruction(complex *sinogram, float wavelength, float n_med, float z0, float dz, float DOF_x, float DOF_z);
complex *odt_reconstruction(complex *sinogram, float wavelength, float n_med, float z0, float dz);
void clean_up();

/*---------------------------------------------------------------------------------*/



int main(int argc, char *argv[])
{
	//float DOF_x, DOF_z = 0;

	// if (argc == 1)
	// {
	// 	printf("No arguments passed, assuming DOF to be at the center!\n");
	// }
	// else if (argc == 3)
	// {
	// 	DOF_x = atof(argv[1]);
	// 	DOF_z = atof(argv[2]);
	// }
	// else
	// {
	// 	fprintf(stderr, "Wrong number of arguments given!\n");
	// }

	if (argc != 1)
	{
		fprintf(stderr, "Wrong number of arguments given!\n");
	}

	printf("Started loading the sinogram.\n");
	complex *sinogram = load_sinogram("sinogram_config.dat", "sinogram.dat");
	printf("Succesfully loaded the sinogram.\n");

	if (Nx == 0 || Ny == 0 || Nx != Ny)
	{
		fprintf(stderr, "Nx or Ny doesn't have an assigned value or are not equal!\n");
		exit(1);
	}
	else
	{
		if (Nx > 512)
		{
			THREADS_PER_BLOCK = 512;
			printf("Set THREADS_PER_BLOCK to 512.\n");
		}
		else
		{
			THREADS_PER_BLOCK = Nx;
			//THREADS_PER_BLOCK = 512;
			printf("Set THREADS_PER_BLOCK to %i.\n", THREADS_PER_BLOCK);
		}
		BLOCKS = cuda::std::ceil((Nx * Ny) / THREADS_PER_BLOCK);
		printf("Set BLOCKS to %i.\n", BLOCKS);
	}
	
	//complex *fft = odt_reconstruction(sinogram, 633E-9, 1, 633E-9, dz, DOF_x, DOF_z);
	complex *fft = odt_reconstruction(sinogram, 633E-9, 1, 633E-9, dz);

	// Free all memory
	clean_up();

	return 0;
}

/*
	Load sinogram, both sinogram_config.dat and sinogram.dat

	Arguments:
		const char *config_file:  Sinogram configuration file
		const char *data_file:    Sinogram data file
*/
complex *load_sinogram(const char *config_file, const char *data_file)
{
	FILE *fptr;

	fptr = fopen(config_file, "r");

	if (fptr == NULL)
	{
		fprintf(stderr, "Failed to open the sinogram data configuration file!\n");
		exit(1);
	}

	fscanf(fptr, "Nx: %i\n", &Nx);
	fscanf(fptr, "Ny: %i\n", &Ny);
	fscanf(fptr, "dx: %f\n", &dx);
	fscanf(fptr, "dy: %f\n", &dy);
	fscanf(fptr, "dz: %f\n", &dz);
	fscanf(fptr, "nproj: %i\n", &nproj);

	if (Nx == 0 || Ny == 0 || nproj == 0 || dx == 0 || dy == 0 || dz == 0)
	{
		fprintf(stderr, "Failed to read sinogram data properties!\n");
		exit(1);
	}

	// Inherently required by ODT reconstruction
	Nz = Nx;

	sinogram = new complex[nproj * Ny * Nx];

	if (sinogram == nullptr)
	{
		fprintf(stderr, "Out of memory allocating sinogram!\n");
		exit(1);
	}

	fclose(fptr);

	fptr = fopen(data_file, "r");

	if (fptr == NULL)
	{
		fprintf(stderr, "Failed to open the sinogram data file!\n");
		exit(1);
	}

	for (int i = 0; i < nproj * Ny * Nx; i++)
	{
		float temp_real, temp_imag;
		fscanf(fptr, "%f+i%f\n", &temp_real, &temp_imag);
		sinogram[i] = cuda::std::complex<float>(temp_real, temp_imag);
	}
			
	fclose(fptr);

	return sinogram;
}

void clean_up()
{
	delete[] sinogram;
}

/*
	Rotate backpropagated beam

	Arguments:
		cuComplex *outputData:    Reconstruction domain scattering potential
		bool realPart:    		  Whetever beam is the real or imag part of the scattering potential
		int Nz:					  Number of points in z
		int Ny:					  Number of points in y
		int Nx:					  Number of points in x
		float theta:			  Angle theta (degrees)
		cudaTextureObject_t:	  Backpropagated beam, as texture object
*/
__global__ void rotate_reconstruction(cuComplex *outputData, bool realPart, int Nz, int Ny, int Nx, float theta, cudaTextureObject_t tex)
{

	// Calculate normalized texture coordinates
	unsigned int z = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int x = blockIdx.z * blockDim.z + threadIdx.z;

	float u = (float)x - (float)Nx / 2;
	float v = (float)y - (float)Ny / 2;
	float w = (float)z - (float)Nz / 2;

	// Create rotated coordinates
	float tu = u * cosf(theta) - w * sinf(theta);
	float tv = v;
	float tw = w * cosf(theta) + u * sinf(theta);

	tu /= (float)Nx;
	tv /= (float)Ny;
	tw /= (float)Nz;

	// Fetch from texture memory and write to global memory
	if (realPart)
	{
		outputData[z * Ny * Nx + y * Nx + x] = cuCaddf(outputData[z * Ny * Nx + y * Ny + x], make_cuFloatComplex(tex3D<float>(tex, tu + 0.5f, tv + 0.5f, tw + 0.5f), 0));
	}
	else
	{
		outputData[z * Ny * Nx + y * Nx + x] = cuCaddf(outputData[z * Ny * Nx + y * Ny + x], make_cuFloatComplex(0, tex3D<float>(tex, tu + 0.5f, tv + 0.5f, tw + 0.5f)));
	}
}

/*
	Rotate backpropagated beam

	Arguments:
		cuComplex *device_rotated_projection:    Pointer to output array, outputs backpropagated beam, not yet rotated
		cuComplex *FFT:    		  				 Fourier transformed measurement
		cuComplex *prefactor:					 Prefactor for ODT, containing normalization and ramp filter
		float *device_M:					  	 M (Angular propagator)
		float dx:								 Delta x
		float dy:								 Delta y
		int Nz:					  				 Number of points in z
		int current_z:							 Index of z value
		int current_proj:			  			 Index of current projection / angle
		float km_px:	  						 Wavenumber medium, in pixels
*/
//__global__ void calculate_backpropagation(cuComplex *device_rotated_projection, cuComplex *FFT, cuComplex *prefactor, float *device_M, float dx, float dy, int Nz, int current_z, int current_proj, float km_px, float corrected_propagation_distance)
__global__ void calculate_backpropagation(cuComplex *device_rotated_projection, cuComplex *FFT, cuComplex *prefactor, float *device_M, float dx, float dy, int Nz, int current_z, int current_proj, float km_px)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < Nz * Nz)
	{
		float z_value = current_z * (1 + 1./(Nz-1)) - Nz / 2;

		//complex angular_propagator = cuda::std::exp(I * km_px * (device_M[index] - 1) * (z_value + corrected_propagation_distance));
		complex angular_propagator = cuda::std::exp(I * km_px * (device_M[index] - 1) * z_value);

		device_rotated_projection[current_z*Nz*Nz + index] = cuCdivf(cuCmulf(cuCmulf(FFT[current_proj*Nz*Nz + index], prefactor[index]), make_cuFloatComplex(cuda::std::real(angular_propagator), cuda::std::imag(angular_propagator))), make_cuFloatComplex(Nz * Nz * dx * dy, 0));
	}
}

/*
	Splits complex array in two seperate arrays for their respective real and imaginairy parts

	Arguments:
		cufftComplex *input:    Pointer to input array
		float *real:    		Pointer to output real part array
		float *imag:			Pointer to output imaginairy part array
		int size:				Size of the array
*/
__global__ void split_complex_array(cufftComplex *input, float *real, float *imag, int size)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < size*size)
	{
		for (int i = 0; i < size; i++)
		{
			real[index + i * size * size] = cuCrealf(input[index + i * size * size]);
			imag[index + i * size * size] = cuCimagf(input[index + i * size * size]);
		}
	}
}

/*
	Here rotation around y-axis is assumed.

	For now Nx & Ny need to be a power of 2.

	Arguments:
		float *sinogram:  3D array of  nproj x Nx x Ny
		float wavelength: vacuum wavelength [m]
		float n_med:      refractive index of medium [-]
		float z0:         detector distance [m]
		float dz:		  delta z [1/m]
*/
//complex *odt_reconstruction(complex *sinogram, float wavelength, float n_med, float z0, float dz, float DOF_x, float DOF_z)
complex *odt_reconstruction(complex *sinogram, float wavelength, float n_med, float z0, float dz)
{
/*---------------------------------------------------------------------------------*/
	// Check input parameters
	if (Nx != Ny)
	{
		fprintf(stderr, "Nx should be equal to Ny!\n");
		exit(1);
	}
	else if ((Nx == 0) || ((Nx & (Nx - 1)) != 0))
	{
		fprintf(stderr, "Nx & Ny should be non-zero and a power of 2!\n");
		exit(1);
	}
	else if (dx == 0 || dy == 0 || dz == 0)
	{
		fprintf(stderr, "dx, dy or dz, don't have an assigned value!\n");
		exit(1);
	}
	else if (sinogram == nullptr)
	{
		fprintf(stderr, "Sinogram is not allocated!\n");
		exit(1);
	}
/*---------------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------------*/
	// Setup variables
	float lambda_px = wavelength / dz;
	float lD = z0 / dz;
	
	float km_px = (2 * M_PI * n_med) / lambda_px;

	float *kx_px = new float[Nx];
	float *ky_px = new float[Ny];

	// This works because we assumed Nx=Ny and a power of two
	for (int i = 0; i < Nx/2; i++)
	{
		kx_px[i] = 2 * M_PI * i / Nx;
		ky_px[i] = 2 * M_PI * i / Ny;
	}
	int temp_count = Nx/2;
	for (int i = -Nx/2; i < 0; i++)
	{
		kx_px[temp_count] = 2 * M_PI * i / Nx;
		ky_px[temp_count] = 2 * M_PI * i / Ny;
		temp_count++;
	}

	float *kxx_px = new float[Nx * Ny];
	float *kyy_px = new float[Nx * Ny];

	for (int i = 0; i < Ny; i++)
	{
		for (int j = 0; j < Nx; j++)
		{
			kxx_px[j + i*Ny] = kx_px[j];
			kyy_px[j + i*Ny] = ky_px[i];
		}
	}

	delete[] kx_px;
    delete[] ky_px;

	bool *filter_klp_px = new bool[Nx * Ny];
	float *M = new float[Nx * Ny];
	complex *prefactor = new complex[Nx * Ny];

	// Weird thing where there is no implementation for *operator
	const float compiler_stupidity = -1;

	for (int i = 0; i < Nx*Ny; i++)
	{
		filter_klp_px[i] = (kxx_px[i]*kxx_px[i] + kyy_px[i]*kyy_px[i] < km_px*km_px);

		M[i] = cuda::std::sqrtf(filter_klp_px[i] * (km_px*km_px - kxx_px[i]*kxx_px[i] - kyy_px[i]*kyy_px[i])) / km_px;

		prefactor[i] = compiler_stupidity*I * km_px * cuda::std::fabs(kxx_px[i]) * static_cast<float>(filter_klp_px[i]) * cuda::std::exp(compiler_stupidity*I * km_px * (M[i] - 1) * lD) / static_cast<float>(nproj);
	}

	float *device_M = nullptr;
	cuComplex *device_prefactor = nullptr;

	cudaCheckError(cudaMalloc((void **)&device_M, Ny * Nx * sizeof(float)));
	cudaCheckError(cudaMalloc((void **)&device_prefactor, Ny * Nx * sizeof(complex)));
	cudaCheckError(cudaMemcpy(device_M, M, Ny * Nx * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(device_prefactor, prefactor, Ny * Nx * sizeof(complex), cudaMemcpyHostToDevice));

	// float DOF_r = cuda::std::sqrtf(DOF_x*DOF_x + DOF_z*DOF_z);
	// float DOF_angle = cuda::std::atan2(DOF_z, DOF_x);

	cuComplex *device_objectFD = nullptr;

	cudaCheckError(cudaMalloc((void **)&device_objectFD, Nz * Ny * Nx * sizeof(cuComplex)));
/*---------------------------------------------------------------------------------*/

	clock_t t; 
    t = clock(); 

/*---------------------------------------------------------------------------------*/
	// FFT for input sinogram
	complex *sinogramFFT = nullptr;
	cuComplex *device_sinogram = nullptr;
	cuComplex *device_sinogramFFT = nullptr;

	sinogramFFT = new complex[nproj * Ny * Nx];

	cudaCheckError(cudaMalloc((void **)&device_sinogram, nproj * Ny * Nx * sizeof(complex)));
	cudaCheckError(cudaMalloc((void **)&device_sinogramFFT, nproj * Ny * Nx * sizeof(complex)));
	cudaCheckError(cudaMemcpy(device_sinogram, sinogram, nproj * Ny * Nx * sizeof(complex), cudaMemcpyHostToDevice));

	// Create cuFFT plan
	cufftHandle fftPlanFwd;
	cufftHandle fftPlanInv;
	
	/*
	Parameters:
		See: https://docs.nvidia.com/cuda/cufft/index.html#cufftmakeplanmany
	*/
	int batch_fwd = nproj;
	int batch_inv = Nz;
   int rank = 2;

	int n[2] = {Ny, Nx};

	int idist = Ny*Nx;
	int odist = Ny*Nx;

	int inembed[2] = {Ny, Nx};
	int onembed[2] = {Ny, Nx};

	int istride = 1;
	int ostride = 1;

	cufftSafeCall(cufftPlanMany(&fftPlanFwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch_fwd));
	cufftSafeCall(cufftPlanMany(&fftPlanInv, rank, n, onembed, ostride, odist, inembed, istride, idist, CUFFT_C2C, batch_inv));

	// Execute cuFFT plan
	cufftSafeCall(cufftExecC2C(fftPlanFwd, (cufftComplex *)device_sinogram, (cufftComplex *)device_sinogramFFT, CUFFT_FORWARD));
	cufftDestroy(fftPlanFwd);

	cudaCheckError(cudaMemcpy(sinogramFFT, device_sinogramFFT, nproj * Ny * Nx * sizeof(complex), cudaMemcpyDeviceToHost));
/*---------------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------------*/
	for (int current_proj = 0; current_proj < nproj; current_proj++)
	{
		printf("Current projection: %i/%i\n", current_proj, nproj);

        float current_angle = 2 * M_PI * current_proj / (nproj - 1);

        //float corrected_propagation_distance = -2 * DOF_r * cuda::std::sinf(current_angle + DOF_angle) / dz;

        cuComplex *device_rotated_projection = nullptr;

        cudaCheckError(cudaMalloc((void **)&device_rotated_projection, Nz * Ny * Nx * sizeof(complex)));

        // Perform all the backpropagations
        for (int current_z = 0; current_z < Nz; current_z++)
        {
      	   //calculate_backpropagation<<<BLOCKS, THREADS_PER_BLOCK>>>(device_rotated_projection, device_sinogramFFT, device_prefactor, device_M, dx, dy, Nz, current_z, current_proj, km_px, corrected_propagation_distance);
      	   calculate_backpropagation<<<BLOCKS, THREADS_PER_BLOCK>>>(device_rotated_projection, device_sinogramFFT, device_prefactor, device_M, dx, dy, Nz, current_z, current_proj, km_px);
        }

        // Allocate memory for the FFT result on the GPU
        cuComplex *device_rotated_projection_invFFT = nullptr;
        cudaCheckError(cudaMalloc((void **)&device_rotated_projection_invFFT, Nz * Ny * Nx * sizeof(complex)));

        // Execute the inverse Fourier transform and free the input array
        cufftSafeCall(cufftExecC2C(fftPlanInv, (cufftComplex *)device_rotated_projection, (cufftComplex *)device_rotated_projection_invFFT, CUFFT_INVERSE)); //CUFFT_INVERSE
		cudaFree(device_rotated_projection);

		float *device_rotated_projection_invFFT_real = nullptr;
		float *device_rotated_projection_invFFT_imag = nullptr;

		cudaCheckError(cudaMalloc((void **)&device_rotated_projection_invFFT_real, Nz * Ny * Nx * sizeof(float)));
		cudaCheckError(cudaMalloc((void **)&device_rotated_projection_invFFT_imag, Nz * Ny * Nx * sizeof(float)));

		split_complex_array<<<BLOCKS, THREADS_PER_BLOCK>>>(device_rotated_projection_invFFT, device_rotated_projection_invFFT_real, device_rotated_projection_invFFT_imag, Nz);
		cudaFree(device_rotated_projection_invFFT);

   /*---------------------------------------------------------------------------------*/

		dim3 dimBlock(8, 8, 8);
	  	dim3 dimGrid(Nz / dimBlock.x, Ny / dimBlock.y, Nx / dimBlock.z);

		// Real Part
		/*---------------------------------------------------------------------------------*/
		//cudaArray Descriptor
		cudaChannelFormatDesc channelDesc_real = cudaCreateChannelDesc<float>();
		//cuda Array
		cudaArray *device_cuArr_real;
		cudaCheckError(cudaMalloc3DArray(&device_cuArr_real, &channelDesc_real, make_cudaExtent(Nz, Ny, Nx), 0));
		cudaMemcpy3DParms copyParams_real = {0};

		//Array creation
		copyParams_real.srcPtr   = make_cudaPitchedPtr(device_rotated_projection_invFFT_real, Nz*sizeof(float), Ny, Nx);
		copyParams_real.dstArray = device_cuArr_real;
		copyParams_real.extent   = make_cudaExtent(Nz, Ny, Nx);
		copyParams_real.kind     = cudaMemcpyDeviceToDevice;
		cudaCheckError(cudaMemcpy3D(&copyParams_real));
		//Array creation End

		cudaResourceDesc    texRes_real;
		memset(&texRes_real, 0, sizeof(cudaResourceDesc));
		texRes_real.resType = cudaResourceTypeArray;
		texRes_real.res.array.array  = device_cuArr_real;
		cudaTextureDesc     texDescr_real;
		memset(&texDescr_real, 0, sizeof(cudaTextureDesc));
		texDescr_real.normalizedCoords = true;
		texDescr_real.filterMode = cudaFilterModeLinear;
		texDescr_real.addressMode[0] = cudaAddressModeClamp;   // clamp
		texDescr_real.addressMode[1] = cudaAddressModeClamp;
		texDescr_real.addressMode[2] = cudaAddressModeClamp;
		texDescr_real.readMode = cudaReadModeElementType;

		// Create the texture so it can be rotated
		static cudaTextureObject_t texReconstruction_real;
		cudaCheckError(cudaCreateTextureObject(&texReconstruction_real, &texRes_real, &texDescr_real, NULL));

	  	rotate_reconstruction<<<dimGrid, dimBlock>>>(device_objectFD, true, Nz, Ny, Nx, current_angle, texReconstruction_real);
	  	/*---------------------------------------------------------------------------------*/

	  	// Imaginairy part
	  	/*---------------------------------------------------------------------------------*/
		//cudaArray Descriptor
		cudaChannelFormatDesc channelDesc_imag = cudaCreateChannelDesc<float>();
		//cuda Array
		cudaArray *device_cuArr_imag;
		cudaCheckError(cudaMalloc3DArray(&device_cuArr_imag, &channelDesc_imag, make_cudaExtent(Nz, Ny, Nx), 0));
		cudaMemcpy3DParms copyParams_imag = {0};

		//Array creation
		copyParams_imag.srcPtr   = make_cudaPitchedPtr(device_rotated_projection_invFFT_imag, Nz*sizeof(float), Ny, Nx);
		copyParams_imag.dstArray = device_cuArr_imag;
		copyParams_imag.extent   = make_cudaExtent(Nz, Ny, Nx);
		copyParams_imag.kind     = cudaMemcpyDeviceToDevice;
		cudaCheckError(cudaMemcpy3D(&copyParams_imag));
		//Array creation End

		cudaResourceDesc    texRes_imag;
		memset(&texRes_imag, 0, sizeof(cudaResourceDesc));
		texRes_imag.resType = cudaResourceTypeArray;
		texRes_imag.res.array.array  = device_cuArr_imag;
		cudaTextureDesc     texDescr_imag;
		memset(&texDescr_imag, 0, sizeof(cudaTextureDesc));
		texDescr_imag.normalizedCoords = true;
		texDescr_imag.filterMode = cudaFilterModeLinear;
		texDescr_imag.addressMode[0] = cudaAddressModeClamp;   // clamp
		texDescr_imag.addressMode[1] = cudaAddressModeClamp;
		texDescr_imag.addressMode[2] = cudaAddressModeClamp;
		texDescr_imag.readMode = cudaReadModeElementType;

		// Create the texture so it can be rotated
		static cudaTextureObject_t texReconstruction_imag;
		cudaCheckError(cudaCreateTextureObject(&texReconstruction_imag, &texRes_imag, &texDescr_imag, NULL));

	  	rotate_reconstruction<<<dimGrid, dimBlock>>>(device_objectFD, false, Nz, Ny, Nx, current_angle, texReconstruction_imag);
	  	/*---------------------------------------------------------------------------------*/

	  	cudaFree(device_rotated_projection_invFFT_real);
	  	cudaFree(device_rotated_projection_invFFT_imag);
	  	cudaFreeArray(device_cuArr_real);
	  	cudaFreeArray(device_cuArr_imag);
	/*---------------------------------------------------------------------------------*/
	
	}
	
	cufftDestroy(fftPlanInv);

	complex* ri = new complex[Nz * Ny * Nx];

	cudaCheckError(cudaMemcpy(ri, device_objectFD, Nz * Ny * Nx * sizeof(cuComplex), cudaMemcpyDeviceToHost));
	cudaFree(device_objectFD);
/*---------------------------------------------------------------------------------*/

	t = clock() - t; 
   double time_taken = ((double)t)/CLOCKS_PER_SEC;

   printf("Finished reconstruction of %i angles in %f seconds!\n", nproj, time_taken);

	FILE *f;

	f = fopen("ri.dat", "w");

	for (int i = 0; i < Nz * Ny * Nx; i++)
	{
		fprintf(f, "%f + i%f\n", n_med*sqrtf(cuda::std::real(ri[i]) / (km_px * km_px / (dz * dz)) + 1), n_med*sqrtf(cuda::std::imag(ri[i]) / (km_px * km_px / (dz * dz)) + 1));
	}

	fclose(f);

   cudaFree(device_sinogram);
   cudaFree(device_sinogramFFT);
   cudaFree(device_M);
   cudaFree(device_prefactor);

   delete[] sinogramFFT;
   delete[] kxx_px;
   delete[] kyy_px;
   delete[] filter_klp_px;
   delete[] M;
   delete[] prefactor;

   return sinogramFFT;
}