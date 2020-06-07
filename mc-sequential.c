#include <sys/time.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define SAMPLE_SIZE (50000000)

typedef struct point {
    float x;
    float y;
} point_t;

uint32_t get_time_in_msec(const struct timeval *time) {
    uint32_t time_msec = time->tv_usec / 1000;
    time_msec += time->tv_sec * 1000;
    return time_msec;}


int main(void)
{
    point_t *points = (point_t *) malloc(sizeof(point_t) * SAMPLE_SIZE);
    char *distances = (char *) malloc(sizeof(char) * SAMPLE_SIZE);

    // Constant seed for reproducibility
    srand(2342);

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        points[i].x = (float) rand() / (float) RAND_MAX;
        points[i].y = (float) rand() / (float) RAND_MAX;
    }

    struct timeval start_time;
    if (gettimeofday(&start_time, NULL) == -1) {
        perror("An error occurred while retrieving a timestamp, exiting");
        exit(-1);
    }

    for (int i = 0; i < SAMPLE_SIZE; i++) {
       float distance = sqrt(points[i].x * points[i].x + points[i].y * points[i].y);

       if (distance <= 1.0) {
           distances[i] = 1;
       } else {
           distances[i] = 0;
       }
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
        n_inside_unit_circle += distances[i];
    }

    float pi = 4 * ((float) n_inside_unit_circle / (float) SAMPLE_SIZE);

    uint32_t msec = get_time_in_msec(&time_diff);

    printf("Computed pi = %f with a sample size of %d in %u msec\n", pi, SAMPLE_SIZE, msec);
}
