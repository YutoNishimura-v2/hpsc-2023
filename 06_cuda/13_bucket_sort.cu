#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cooperative_groups.h>
using namespace cooperative_groups;


__device__ void init_bucket(int *bucket, int i, int range){
  if(i>=range) return;
  bucket[i] = 0;
}


__device__ void bucket_update(int *bucket, int *key, int i, int n){
  if(i>=n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__device__ void bucket_scan(int *bucket, int *bucket_sub, int i, int range){
  if(i>=range) return;
  grid_group grid = this_grid();
  for(int j=1; j<range; j<<=1) {
    bucket_sub[i] = bucket[i];
    grid.sync();
    if(i>=j) bucket[i] += bucket_sub[i-j];
    grid.sync();
  }
}

__device__ int binary_search(int *bucket, int i, int range){
  if(bucket[0]>i) return 0;
  int r = 0;
  int l = range;
  int m;
  while(l - r > 1){
    m = (l + r) / 2;
    if(bucket[m] > i){
      l = m;
    }else{
      r = m;
    }
  }
  return l;
}

__device__ void key_update(int *bucket, int *key, int i, int n, int range){
  if(i>=n) return;
  key[i] = binary_search(bucket, i, range);
}

__global__ void bucketsort(
  int *key, int *bucket, int *bucket_sub, int n, int range
){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  grid_group grid = this_grid();
  init_bucket(bucket, i, range);
  grid.sync();
  bucket_update(bucket, key, i, n);
  grid.sync();
  bucket_scan(bucket, bucket_sub, i, range);
  key_update(bucket, key, i, n, range);
  grid.sync();
}

int main() {
  int n = 50;
  int range = 5;
  const int M = 64;
  int *key, *bucket, *bucket_sub;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMalloc(&bucket_sub, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");
  void *args[] = {
    (void *)&key,  (void *)&bucket, (void *)&bucket_sub, (void *)&n, (void *)&range
  };
  cudaLaunchCooperativeKernel(
    (void*)bucketsort, (n+M-1)/M, M, args
  );
  cudaDeviceSynchronize();

  for (int i=0; i<range; i++) {
    printf("%d ",bucket[i]);
  }
  printf("\n");
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  cudaFree(key);
  cudaFree(bucket);
  cudaFree(bucket_sub);
}
