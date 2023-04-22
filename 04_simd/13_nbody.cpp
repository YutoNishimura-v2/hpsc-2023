#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

float sum_vec(__m256 avec, int N){
  float output[N];
  __m256 bvec = _mm256_permute2f128_ps(avec,avec,1);
  bvec = _mm256_add_ps(bvec,avec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  bvec = _mm256_hadd_ps(bvec,bvec);
  _mm256_store_ps(output, bvec);
  return output[0];
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);

    __m256 mask = _mm256_cmp_ps(xivec, xvec, _CMP_NEQ_OQ);

    __m256 r_twovec = _mm256_mul_ps(rxvec, rxvec) + _mm256_mul_ps(ryvec, ryvec);
    __m256 r_inv_threevec = _mm256_div_ps(_mm256_rsqrt_ps(r_twovec), r_twovec);

    __m256 masked_r_inv_threevec = _mm256_blendv_ps(_mm256_setzero_ps(), r_inv_threevec, mask);

    __m256 fx_i_vec = _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec), masked_r_inv_threevec);
    __m256 fy_i_vec = _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec), masked_r_inv_threevec);
    fx[i] -= sum_vec(fx_i_vec, N);
    fy[i] -= sum_vec(fy_i_vec, N);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
