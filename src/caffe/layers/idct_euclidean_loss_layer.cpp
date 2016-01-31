#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

const int zigzag_lookup[8][8] = { 
{ 0, 1, 5, 6, 14, 15, 27, 28 },
{ 2, 4, 7, 13, 16, 26, 29, 42},
{ 3, 8, 12, 17, 25, 30, 41, 43},
{ 9, 11, 18, 24, 31, 40, 44, 53},
{ 10, 19, 23, 32, 39, 45, 52, 54},
{ 20, 22, 33, 38, 46, 51, 55, 60},
{ 21, 34, 37, 47, 50, 56, 59, 61},
{ 35, 36, 48, 49, 57, 58, 62, 63} 
};

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::computeDIdy(Dtype* derivs)
{
  Dtype a0 = (Dtype)(sqrt(2.0 / 8.0));
  Dtype a1 = (Dtype)(1.0 / sqrt(8.0));
  for (int m = 0; m < 8; ++m) {
    for (int n = 0; n < 8; ++n) {
      int i = m * 8 + n;
      for (int p = 0; p < 8; ++p) {
        for (int q = 0; q < 8; ++q) {
          int j = p * 8 + q;
          Dtype a_p = a0;
          Dtype a_q = a0;
          if (p == 0) {
            a_p = a1;
          }
          if (q == 0) {
            a_q = a1;
          }
//          (*dIdy)(i, j) = (Dtype)(a_p * a_q * cos(Rd::Math::PI<Dtype>() * (2 * m + 1) * p / 16.0) * cos(Rd::Math::PI<Dtype>() * (2 * n + 1) * q / 16.0));
          derivs[i * 64 + j] = (Dtype)(a_p * a_q * A_cpu[i * 64 + j]);
        }
      }
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::computeIdct2(const Dtype* c_coeffs, Dtype* local_pixels)
{
  const Dtype a0 = (Dtype)(1.0 / (Dtype)sqrt(8.0));
  const Dtype a1 = (Dtype)sqrt((Dtype)(2.0 / 8.0));
  for (int i = 0; i < 64; ++i) {
    local_pixels[i] = 0.0;
    for (int k = 0; k < 64; ++k) {
      int p = k / 8;
      int q = k % 8;
      Dtype a_p = a1; 
      Dtype a_q = a1; 
      if (p == 0) {
        a_p = a0; 
      }
      if (q == 0) {
        a_q = a0; 
      }
      //local_pixels[i] += Dtype(a_p * a_q * c_coeffs[zigzag_lookup[p][q]] * cos(M_PI * p * (2*m+1) / 16.0) * cos(M_PI * q * (2*n+1) / 16.0));
      local_pixels[i] += Dtype(a_p * a_q * c_coeffs[zigzag_lookup[p][q]] * A_cpu[i*64+k]);
      //local_pixels[i] += Dtype(a_p * a_q * c_coeffs[k] * A_cpu[i*64+k]);
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  computeDIdy(idct_derivs_);
  all_pixels0_.ReshapeLike(*bottom[0]);
  all_pixels1_.ReshapeLike(*bottom[0]);
  idct2_derivs_.ReshapeLike(*bottom[0]);
  caffe_gpu_get_didct2(1, idct2_derivs_.mutable_gpu_data());
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
  //idct2_derivs_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  const int block_size = 64;
  const int example_size = block_size * 9;
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
      computeIdct2(c_local_coeffs0, local_pixels0);
      computeIdct2(c_local_coeffs1, local_pixels1);
      /*if (j == 128) {
        std::cout << "coeffs:" << std::endl;
        for (int c = 0; c < 64; ++c) {
          int p = c / 8;
          int q = c % 8;
          std::cout << c_local_coeffs1[zigzag_lookup[p][q]];
          if (q == 7) std::cout << ";\n";
          else std::cout << ",";
        }
        std::cout << "pixels:" << std::endl;
        for (int c = 0; c < 64; ++c) {
          int p = c / 8;
          int q = c % 8;
          std::cout << local_pixels1[c];
          if (q == 7) std::cout << ";\n";
          else std::cout << ",";
        }
      }*/
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
  /*Dtype A[3] = {1., 2., 3.};
  Dtype B[9] = {1., 2., 3., 4., 5., 6., 7., 8., 9.};
  Dtype C[3];
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 3, 3, 1.0, A, B, 0., C); 
  for (int i = 0; i < 3; ++i) {
    std::cout << C[i] << ",";
  }
  std::cout << std::endl;
  LOG(FATAL) << "aergaerg";
  */
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      const int block_size = 64;
      const int example_size = 9 * block_size;
      const int N = bottom[i]->count() / example_size;
      const Dtype* idct_derivs = (const Dtype*)idct_derivs_;
      for (int j = 0; j < N; ++j) {
        // compute dE/dI * dI/dy for each example in the batch
        for (int k = 0; k < 9; ++k) {
          const Dtype* diff = diff_.cpu_data() + j*example_size + k*block_size;
          Dtype* result = bottom[i]->mutable_cpu_diff() + j*example_size + k*block_size;
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 64, 64,
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
