//MonteCarlo 
    //r = 1인 원의 넓이를 몬테카를로 방법을 이용해 추정
    //4 : 원의 넓이 = 전체sampling횟수 : 범위 안에 들어간 sampling 횟수 
#include <iostream>
#include <cstdlib>
#include <cmath>



class MonteCarlo {
        private:
            float x;
            float y;         // -1<=x, y<=1
            float dist;       //원점과의 거리
        public:
            MonteCarlo();               //constructor 
            float ret_dist() {return dist;}
    };

MonteCarlo::MonteCarlo() {
    srand(time(0)* rand() / rand());  
    x = (float)rand()/(float)INT32_MAX;
    y = (float)rand()/(float)INT32_MAX;

    srand(time(0)* rand());
    float sign_prob = (float)rand()/(float)INT32_MAX;
    if (sign_prob <= 0.5) {
        x = -x;
    }
    srand(time(0)* rand()*rand());
    sign_prob = (float)rand()/(float)INT32_MAX;
    if (sign_prob <= 0.5) {
        y = -y;
    }

    dist = sqrt(((x*x)+(y*y)));
}


int main() {
    using std::cout;
    using std::endl;

    
    int roof_size_1000 = 1000;
    int roof_size_10000 = 10000;
    int roof_size_100000 = 100000;
    /*
    for (int i = 0; i < roof_size_1000) {
        new MonteCarlo();
    }
    */
   
   int ct_inner = 0;

   for (int i = 0; i < roof_size_1000; i++) { 
    MonteCarlo * mc = new MonteCarlo();
    if (mc->ret_dist() <= 1.00)
        ct_inner++;
    delete mc;
    cout << endl << endl;
   }

   float area = (float)(4 * ct_inner) / ((float)roof_size_1000);
   cout << area << endl;

   return 0;
}