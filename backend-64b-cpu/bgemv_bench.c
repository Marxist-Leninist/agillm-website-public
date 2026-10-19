// 1-bit weight x int8-activation GEMV via bit-planes + popcount (claude-chat-64b-improve-20261019, feasibility bench)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
static void bgemv(const uint64_t *W, const int64_t *rowpop, int rows, int n, const float *x, float *y){
  int nw = n / 64; float amax = 1e-12f;
  for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
  float sc = amax / 127.f; uint64_t *U = aligned_alloc(64, 8 * nw * 8); int64_t cntU[8] = {0};
  for (int b = 0; b < 8; b++) for (int w = 0; w < nw; w++) U[b * nw + w] = 0;
  for (int j = 0; j < n; j++) { int q = (int)lrintf(x[j] / sc); if (q > 127) q = 127; if (q < -127) q = -127; unsigned u = (unsigned)(q + 128);
    for (int b = 0; b < 8; b++) if (u >> b & 1u) U[b * nw + j / 64] |= 1ull << (j % 64); }
  for (int b = 0; b < 8; b++) for (int w = 0; w < nw; w++) cntU[b] += __builtin_popcountll(U[b * nw + w]);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < rows; i++) {
    const uint64_t *r = W + (size_t)i * nw; int64_t a[8] = {0};
    for (int w = 0; w < nw; w++) { uint64_t v = r[w];
      a[0] += __builtin_popcountll(v & U[w]);          a[1] += __builtin_popcountll(v & U[nw + w]);
      a[2] += __builtin_popcountll(v & U[2 * nw + w]); a[3] += __builtin_popcountll(v & U[3 * nw + w]);
      a[4] += __builtin_popcountll(v & U[4 * nw + w]); a[5] += __builtin_popcountll(v & U[5 * nw + w]);
      a[6] += __builtin_popcountll(v & U[6 * nw + w]); a[7] += __builtin_popcountll(v & U[7 * nw + w]); }
    int64_t s = 0; for (int b = 0; b < 8; b++) s += (2 * a[b] - cntU[b]) << b;
    s -= 128 * (2 * rowpop[i] - n); y[i] = sc * (float)s; }
  free(U);
}
int main(int argc, char **argv){
  int rows = atoi(argv[1]), n = atoi(argv[2]), reps = atoi(argv[3]); int nw = n / 64;
  uint64_t *W = aligned_alloc(64, (size_t)rows * nw * 8); int64_t *rp = malloc(rows * 8);
  float *x = malloc(n * 4), *y = malloc(rows * 4), *yr = malloc(rows * 4);
  uint64_t s = 88172645463325252ull;
  for (size_t k = 0; k < (size_t)rows * nw; k++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; W[k] = s; }
  for (int i = 0; i < rows; i++) { int64_t c = 0; for (int w = 0; w < nw; w++) c += __builtin_popcountll(W[(size_t)i * nw + w]); rp[i] = c; }
  for (int j = 0; j < n; j++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; x[j] = (float)((int64_t)(s % 2001) - 1000) / 333.f; }
  bgemv(W, rp, rows, n, x, y);
  // reference on 64 rows: exact float dot with +-1 weights, and with the same int8 quantisation
  double maxrel = 0; float amax = 1e-12f; for (int j = 0; j < n; j++) if (fabsf(x[j]) > amax) amax = fabsf(x[j]);
  float sc = amax / 127.f;
  for (int i = 0; i < 64; i++) { double ref = 0, refq = 0; for (int j = 0; j < n; j++) { int bit = W[(size_t)i * nw + j / 64] >> (j % 64) & 1; double sgn = bit ? 1 : -1;
      ref += sgn * x[j]; int q = (int)lrintf(x[j] / sc); if (q > 127) q = 127; if (q < -127) q = -127; refq += sgn * q * sc; }
    double e = fabs(y[i] - refq) / (fabs(refq) + 1e-9); if (e > maxrel) maxrel = e; if (i == 0) printf("row0 y %.4f refq %.4f ref_fp %.4f\n", y[0], refq, ref); }
  printf("max rel err vs int8 ref over 64 rows: %.2e\n", maxrel);
  double t0 = omp_get_wtime(); for (int r = 0; r < reps; r++) bgemv(W, rp, rows, n, x, y); double dt = (omp_get_wtime() - t0) / reps;
  printf("%d x %d bits (%.1f MB): %.2f ms/GEMV, %.1f GB/s of weight bits, threads %d\n", rows, n, rows * (double)n / 8e6, dt * 1e3, rows * (double)n / 8 / dt / 1e9, omp_get_max_threads());
  return 0;
}
