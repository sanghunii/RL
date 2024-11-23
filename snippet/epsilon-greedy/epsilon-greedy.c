//epsilon-greedy algorithm Idea 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main() {
    float epsilon = 0.07;          
    float explore_ratio = 0.0;

    srand(time(0)* rand());         

    int ct_epsilon = 0;
    int roof_size = 10000;

    for (int i = 0; i < roof_size; i++) {
        if ((float)rand()/(float)INT32_MAX <= epsilon)
            ct_epsilon++;
    }

    printf("ratio of exploration : %.2f", (float)ct_epsilon/(float)roof_size);
}