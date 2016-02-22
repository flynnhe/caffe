#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::computeDIdy(Dtype* derivs, const int npix)
{
  const int N = npix;
  const int N2 = N*N;
  Dtype a0 = (Dtype)(sqrt(2.0 / N));
  Dtype a1 = (Dtype)(1.0 / sqrt(N));
  for (int m = 0; m < N; ++m) {
    for (int n = 0; n < N; ++n) {
      int i = m * N + n;
      for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
          int j = p * N + q;
          Dtype a_p = a0;
          Dtype a_q = a0;
          if (p == 0) {
            a_p = a1;
          }
          if (q == 0) {
            a_q = a1;
          }
//          (*dIdy)(i, j) = (Dtype)(a_p * a_q * cos(Rd::Math::PI<Dtype>() * (2 * m + 1) * p / 16.0) * cos(Rd::Math::PI<Dtype>() * (2 * n + 1) * q / 16.0));
          //derivs[i * 64 + j] = (Dtype)(a_p * a_q * A_cpu[i * 64 + j]);
          derivs[i * N2 + j] = (Dtype)(a_p * a_q * cos(M_PI*p*(2*m+1)/(2*N)) * cos(M_PI*q*(2*n+1)/(2*N))); 
        }
      }
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::computeIdct2(const Dtype* c_coeffs, Dtype* local_pixels, const int npix)
{
  const int N = npix;
  const int N2 = N*N;
  const Dtype a0 = (Dtype)(1.0 / (Dtype)sqrt(N));
  const Dtype a1 = (Dtype)sqrt((Dtype)(2.0 / N));
  for (int i = 0; i < N2; ++i) {
    local_pixels[i] = 0.0;
    int m = i / N;
    int n = i % N;
    for (int k = 0; k < N2; ++k) {
      int p = k / N;
      int q = k % N;
      Dtype a_p = a1; 
      Dtype a_q = a1; 
      if (p == 0) {
        a_p = a0; 
      }
      if (q == 0) {
        a_q = a0; 
      }
      //local_pixels[i] += Dtype(a_p * a_q * c_coeffs[zigzag_lookup[p][q]] * cos(M_PI * p * (2*m+1) / 16.0) * cos(M_PI * q * (2*n+1) / 16.0));
      //local_pixels[i] += Dtype(a_p * a_q * c_coeffs[zigzag_lookup[p][q]] * A_cpu[i*64+k]);
      local_pixels[i] += (Dtype)(a_p * a_q * c_coeffs[k] * cos(M_PI * p * (2*m+1) / 16.0) * cos(M_PI * q * (2*n+1) / 16.0));
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  const int npix = 24;
  const int N2 = npix*npix;
  computeDIdy(idct_derivs_, npix);
  all_pixels0_.ReshapeLike(*bottom[0]);
  all_pixels1_.ReshapeLike(*bottom[0]);
  std::vector<int> v;
  v.push_back(bottom[0]->num());
  v.push_back(bottom[0]->channels());
  v.push_back(N2);
  v.push_back(N2);
  idct2_derivs_.Reshape(v);
  caffe_gpu_get_didct2(1, npix, idct2_derivs_.mutable_gpu_data());
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  all_pixels0_.ReshapeLike(*bottom[0]);
  all_pixels1_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  const int num_blocks = 1;
  const int npix = 24;
  const int block_size = npix*npix;
  const int example_size = num_blocks * block_size;
  const Dtype* c_coeffs0 = bottom[0]->cpu_data();
  const Dtype* c_coeffs1 = bottom[1]->cpu_data();

  // compute the 2d idct
  Dtype all_pixels0[count];
  Dtype all_pixels1[count];
  const int N = count / example_size;
  for (int i = 0; i < N; ++i) { // go through each example in the batch
    Dtype* curr_example_pixels0 = all_pixels0 + i*example_size;
    Dtype* curr_example_pixels1 = all_pixels1 + i*example_size;
    const Dtype* c_curr_example_coeffs0 = c_coeffs0 + i*example_size;
    const Dtype* c_curr_example_coeffs1 = c_coeffs1 + i*example_size;
    for (int j = 0; j < example_size; j += block_size) {
      // compute the 2d idct for each block
      Dtype* local_pixels0 = curr_example_pixels0 + j;
      Dtype* local_pixels1 = curr_example_pixels1 + j;
      const Dtype* c_local_coeffs0 = c_curr_example_coeffs0 + j;
      const Dtype* c_local_coeffs1 = c_curr_example_coeffs1 + j;
      computeIdct2(c_local_coeffs0, local_pixels0, npix);
      computeIdct2(c_local_coeffs1, local_pixels1, npix);
    }
  }
  caffe_sub(
      count,
      all_pixels0,
      all_pixels1,
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      const int num_blocks = 1;
      const int npix = 24; // w or h of square
      const int block_size = npix*npix;
      const int example_size = num_blocks * block_size;
      const int N = bottom[i]->count() / example_size;
      const Dtype* idct_derivs = (const Dtype*)idct_derivs_;
      for (int j = 0; j < N; ++j) {
        // compute dE/dI * dI/dy for each example in the batch
        for (int k = 0; k < num_blocks; ++k) {
          const Dtype* diff = diff_.cpu_data() + j*example_size + k*block_size;
          Dtype* result = bottom[i]->mutable_cpu_diff() + j*example_size + k*block_size;
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, block_size, block_size,
                                alpha, diff, idct_derivs, 0., result);
        }
       }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(IdctEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(IdctEuclideanLossLayer);
REGISTER_LAYER_CLASS(IdctEuclideanLoss);

}  // namespace caffe
