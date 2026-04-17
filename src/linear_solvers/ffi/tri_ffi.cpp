#include "xla/ffi/api/ffi.h"
#include <cstdlib>

namespace ffi = xla::ffi;

// Tridiagonal Thomas algorithm.
//   a[i] = A[i+1, i]  (lower diagonal, length n-1)
//   b[i] = A[i, i]    (main diagonal,  length n)
//   c[i] = A[i, i+1]  (upper diagonal, length n-1)
//   d[i]               (RHS,            length n)
static void tri_thomas_f64(const double *a, const double *b, const double *c,
                           const double *d, double *x, int64_t n) {
  double *cm = (double *)malloc(n * sizeof(double));
  double *dm = (double *)malloc(n * sizeof(double));

  // Row 0
  double inv = 1.0 / b[0];
  cm[0] = c[0] * inv;
  dm[0] = d[0] * inv;

  // Rows 1..n-1
  for (int64_t i = 1; i < n; i++) {
    inv = 1.0 / (b[i] - a[i - 1] * cm[i - 1]);
    cm[i] = (i < n - 1 ? c[i] : 0.0) * inv;
    dm[i] = (d[i] - a[i - 1] * dm[i - 1]) * inv;
  }

  // Backward sweep
  x[n - 1] = dm[n - 1];
  for (int64_t i = n - 2; i >= 0; i--)
    x[i] = dm[i] - cm[i] * x[i + 1];

  free(cm);
  free(dm);
}

ffi::Error TriSolveF64(ffi::Buffer<ffi::F64> a_buf, ffi::Buffer<ffi::F64> b_buf,
                       ffi::Buffer<ffi::F64> c_buf, ffi::Buffer<ffi::F64> d_buf,
                       ffi::Result<ffi::Buffer<ffi::F64>> out) {
  int64_t n = b_buf.element_count();
  tri_thomas_f64(a_buf.typed_data(), b_buf.typed_data(), c_buf.typed_data(),
                 d_buf.typed_data(), out->typed_data(), n);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(TriSolveF64FFI, TriSolveF64,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>() // a
                                  .Arg<ffi::Buffer<ffi::F64>>() // b
                                  .Arg<ffi::Buffer<ffi::F64>>() // c
                                  .Arg<ffi::Buffer<ffi::F64>>() // d
                                  .Ret<ffi::Buffer<ffi::F64>>() // x out
);
