#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <string.h>

#define SAMPLE_SIZE (50000000)

typedef struct point {
    float x;
    float y;
} point_t;

uint32_t get_time_in_msec(const struct timeval *time) {
    uint32_t time_msec = time->tv_usec / 1000;
    time_msec += time->tv_sec * 1000;
    return time_msec;
}

static char* read_kernel_source(const char *pathname)
{
    struct stat statbuf;
    if (lstat(pathname, &statbuf) == -1) {
        perror("lstat");
        exit(EXIT_FAILURE);
    }

    FILE *f = fopen(pathname, "r");
    if (f == NULL) {
        perror("Error: Failed to open file with kernel source");
        exit(EXIT_FAILURE);
    }

    char *kernel_source = (char*) malloc(statbuf.st_size);
    size_t n_bytes = fread(kernel_source, 1, statbuf.st_size, f);
    if (n_bytes != (size_t) statbuf.st_size) {
        perror("fread");
        free(kernel_source);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);

    return kernel_source;
}

cl_kernel create_distance_kernel(cl_context context, cl_device_id device_id)
{
    cl_int err;

    char *kernel_source = read_kernel_source("evaluate_point.cl");
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        perror("Error while creating program");
        exit(-1);
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(-1);
    }

    cl_kernel kernel = clCreateKernel(program, "evaluate_point", &err);
    if (err != CL_SUCCESS) {
        perror("Error: Failed to create kernel");
        exit(-1);
    }

    free(kernel_source);
    return kernel;
}

void initialize_context(cl_device_id *device_id, cl_context *context, cl_command_queue *command_queue)
{
    cl_int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, device_id, NULL);
    if (err != CL_SUCCESS) {
        perror("Error while retrieving device ids");
        exit(-1);
    }

    *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        perror("Error while retrieving a OpenCL context");
        exit(-1);
    }

    *command_queue = clCreateCommandQueue(*context, *device_id, 0, &err);
    if (err != CL_SUCCESS) {
        perror("Error while creating command queue");
        exit(-1);
    }
}


int main(void) {

    unsigned int count = SAMPLE_SIZE;

    point_t *points = (point_t *) malloc(sizeof(point_t) * SAMPLE_SIZE);
    char *hits = (char *) malloc(sizeof(char) * SAMPLE_SIZE);

    // Constant seed for reproducibility
    srand(2342);
    
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        points[i].x = (float) rand() / (float) RAND_MAX;
        points[i].y = (float) rand() / (float) RAND_MAX;
    }

    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    initialize_context(&device_id, &context, &command_queue);

    cl_kernel kernel = create_distance_kernel(context, device_id);

    cl_mem input = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(struct point) * count, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * count, NULL, NULL);
    if (!input || !output) {
        perror("Error: Failed to allocate device memory");
        return -1;
    }

    cl_int err;

    err  = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
    if (err != CL_SUCCESS) {
        perror("Error: Failed to set kernel arguments");
        return -1;
    }

    size_t local;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        perror("Error: Failed to get kernel work group info");
        return -1;
    }

    struct timeval start_time;
    if (gettimeofday(&start_time, NULL) == -1) {
        perror("An error occurred while retrieving a timestamp, exiting");
        exit(-1);
    }

    err = clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0, sizeof(struct point) * count, points, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        perror("Error: Failed to enqueue writer buffer");
        return -1;
    }

    size_t global = count;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        perror("Error: Failed to execute kernel");
        return -1;
    }

    clFinish(command_queue);

    err = clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, sizeof(char) * count, hits, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        perror("Error: Failed to read results buffer from device");
        return -1;
    }

    struct timeval end_time;
    if (gettimeofday(&end_time, NULL) == -1) {
        perror("An error occurred while retrieving a timestamp, exiting");
        exit(-1);
    }

    struct timeval time_diff;
    timersub(&end_time, &start_time, &time_diff);

    int n_inside_unit_circle = 0;
    for (int i = 0; i < SAMPLE_SIZE; i++) {
        n_inside_unit_circle += hits[i];
    }

    float pi = 4 * ((float) n_inside_unit_circle / (float) SAMPLE_SIZE);

    uint32_t msec = get_time_in_msec(&time_diff);

    printf("Computed pi = %f with a sample size of %d in %u msec\n", pi, SAMPLE_SIZE, msec);

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    return 0;
}
