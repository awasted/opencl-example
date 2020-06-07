
typedef struct point {
    float x;
    float y;
} point_t;

__kernel void evaluate_point(__global point_t* points,
                             __global char* hits,
                             const unsigned int count)
{
   int i = get_global_id(0);
   if (i < count) {
       float distance = sqrt(points[i].x * points[i].x + points[i].y * points[i].y);

       if (distance <= 1.0) {
           hits[i] = 1;
       } else {
           hits[i] = 0;
       }
   }
}
