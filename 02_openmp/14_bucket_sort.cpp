#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  // init
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  #pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++)
    #pragma omp atomic update
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  // parallelize prefix sum
  std::vector<int> _offset(range,0);
  // enabling parallel for loop
  #pragma omp parallel
  for (int j=1; j<range; j<<=1){
    #pragma omp for
    for (int i=0; i<range; i++){
      _offset[i] = offset[i];
    }
    #pragma omp for
    for (int i=j; i<range; i++){
      offset[i] += _offset[i-j] + bucket[i-j];
    }
  }
  // parallelize key insertation
  #pragma omp parallel for schedule(dynamic, 1)
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (int k=bucket[i]; k>0; k--) {
      key[j+k-1] = i;
    }
  }
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
