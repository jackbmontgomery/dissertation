#include "xla/ffi/api/ffi.h"
#include <cstdlib>
#include <cstring>

namespace ffi = xla::ffi;

// Pentadiagonal Thomas algorithm
// Convention:
//   e[i] = A[i+2, i]  (second subdiagonal,  length n-2)
//   a[i] = A[i+1, i]  (first subdiagonal,   length n-1)
//   b[i] = A[i, i]    (main diagonal,        length n)
//   c[i] = A[i, i+1]  (first superdiagonal,  length n-1)
//   f[i] = A[i, i+2]  (second superdiagonal, length n-2)
//   d[i]               (RHS,                  length n)
static void penta_thomas_f64(const double *e, const double *a, const double *b,
                             const double *c, const double *f, const double *d,
                             double *x, int64_t n) {
  double *cm = (double *)malloc(n * sizeof(double));
  double *fm = (double *)malloc(n * sizeof(double));
  double *dm = (double *)malloc(n * sizeof(double));

  // Row 0
  double inv = 1.0 / b[0];
  cm[0] = c[0] * inv;
  fm[0] = f[0] * inv;
  dm[0] = d[0] * inv;

  // Row 1
  inv = 1.0 / (b[1] - a[0] * cm[0]);
  cm[1] = (c[1] - a[0] * fm[0]) * inv;
  fm[1] = (n > 3 ? f[1] : 0.0) * inv;
  dm[1] = (d[1] - a[0] * dm[0]) * inv;

  // Rows 2..n-1
  for (int64_t i = 2; i < n; i++) {
    double c_val = (i < n - 1) ? c[i] : 0.0;
    double f_val = (i < n - 2) ? f[i] : 0.0;
    double alpha = a[i - 1] - e[i - 2] * cm[i - 2];
    inv = 1.0 / (b[i] - e[i - 2] * fm[i - 2] - alpha * cm[i - 1]);
    cm[i] = (c_val - alpha * fm[i - 1]) * inv;
    fm[i] = f_val * inv;
    dm[i] = (d[i] - e[i - 2] * dm[i - 2] - alpha * dm[i - 1]) * inv;
  }

  // Backward sweep
  x[n - 1] = dm[n - 1];
  x[n - 2] = dm[n - 2] - cm[n - 2] * x[n - 1];
  for (int64_t i = n - 3; i >= 0; i--)
    x[i] = dm[i] - cm[i] * x[i + 1] - fm[i] * x[i + 2];

  free(cm);
  free(fm);
  free(dm);
}

ffi::Error
PentaSolveF64(ffi::Buffer<ffi::F64> e_buf, ffi::Buffer<ffi::F64> a_buf,
              ffi::Buffer<ffi::F64> b_buf, ffi::Buffer<ffi::F64> c_buf,
              ffi::Buffer<ffi::F64> f_buf, ffi::Buffer<ffi::F64> d_buf,
              ffi::Result<ffi::Buffer<ffi::F64>> out) {
  int64_t n = b_buf.element_count();
  penta_thomas_f64(e_buf.typed_data(), a_buf.typed_data(), b_buf.typed_data(),
                   c_buf.typed_data(), f_buf.typed_data(), d_buf.typed_data(),
                   out->typed_data(), n);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PentaSolveF64FFI, PentaSolveF64,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>() // e
                                  .Arg<ffi::Buffer<ffi::F64>>() // a
                                  .Arg<ffi::Buffer<ffi::F64>>() // b
                                  .Arg<ffi::Buffer<ffi::F64>>() // c
                                  .Arg<ffi::Buffer<ffi::F64>>() // f
                                  .Arg<ffi::Buffer<ffi::F64>>() // d
                                  .Ret<ffi::Buffer<ffi::F64>>() // x out
);
