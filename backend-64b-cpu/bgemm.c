// libbgemm: Y[N,rows] = X[N,n] @ W^T for W in {-1,+1} packed 1 bit/weight (bit 1 = +1), little-endian per uint64 word.
// Activations quantised per row to int8 (absmax/127, the BitNet A8 rule); 8 bit-planes + popcount; exact vs the
// int8 reference. claude-chat-64b-improve-20261019.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
void bgemm(const uint64_t *W, const int64_t *rowpop, int rows, int n, const float *X, int N, float *Y) {
  const int nw = n / 64;
  uint64_t *U = (uint64_t *)aligned_alloc(64, (size_t)N * 8 * nw * 8 + 64);
  int64_t *cnt = (int64_t *)malloc((size_t)N * 8 * sizeof(int64_t));
  float *sc = (float *)malloc((size_t)N * sizeof(float));
  memset(U, 0, (size_t)N * 8 * nw * 8);
  #pragma omp parallel for schedule(static)
  for (int t = 0; t < N; t++) {
    const float *x = X + (size_t)t * n; float amax = 1e-30f;
    for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
    float s = amax / 127.f; sc[t] = s; uint64_t *P = U + (size_t)t * 8 * nw;
    for (int j = 0; j < n; j++) { int q = (int)lrintf(x[j] / s); q = q > 127 ? 127 : (q < -127 ? -127 : q);
      unsigned u = (unsigned)(q + 128); for (int b = 0; b < 8; b++) if (u >> b & 1u) P[b * nw + (j >> 6)] |= 1ull << (j & 63); }
    for (int b = 0; b < 8; b++) { int64_t c = 0; for (int w = 0; w < nw; w++) c += __builtin_popcountll(P[b * nw + w]); cnt[t * 8 + b] = c; }
  }
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < rows; i++) {
    const uint64_t *r = W + (size_t)i * nw; const int64_t base = 128 * (2 * rowpop[i] - n);
    for (int t = 0; t < N; t++) {
      const uint64_t *P = U + (size_t)t * 8 * nw; int64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
      for (int w = 0; w < nw; w++) { const uint64_t v = r[w];
        a0 += __builtin_popcountll(v & P[w]);          a1 += __builtin_popcountll(v & P[nw + w]);
        a2 += __builtin_popcountll(v & P[2 * nw + w]); a3 += __builtin_popcountll(v & P[3 * nw + w]);
        a4 += __builtin_popcountll(v & P[4 * nw + w]); a5 += __builtin_popcountll(v & P[5 * nw + w]);
        a6 += __builtin_popcountll(v & P[6 * nw + w]); a7 += __builtin_popcountll(v & P[7 * nw + w]); }
      const int64_t *c = cnt + t * 8;
      int64_t s = (2 * a0 - c[0]) + ((2 * a1 - c[1]) << 1) + ((2 * a2 - c[2]) << 2) + ((2 * a3 - c[3]) << 3)
                + ((2 * a4 - c[4]) << 4) + ((2 * a5 - c[5]) << 5) + ((2 * a6 - c[6]) << 6) + ((2 * a7 - c[7]) << 7);
      Y[(size_t)t * rows + i] = sc[t] * (float)(s - base);
    }
  }
  free(U); free(cnt); free(sc);
}
