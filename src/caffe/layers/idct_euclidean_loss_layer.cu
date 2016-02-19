#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iomanip>

namespace caffe {

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const int num_blocks = 9;
  const int block_size = 64;
  const int example_size = block_size * num_blocks;

  const Dtype* c_coeffs0 = bottom[0]->gpu_data();
  const Dtype* c_coeffs1 = bottom[1]->gpu_data();

  // compute the 2d idct
  Dtype* all_pixels0 = all_pixels0_.mutable_gpu_data();
  Dtype* all_pixels1 = all_pixels1_.mutable_gpu_data();

  const int num_examples = count / example_size;

  // compute the idct2 bearing in mind the coeffs are in zigzag format
  caffe_gpu_idct2(num_examples, c_coeffs0, c_coeffs1, all_pixels0, all_pixels1);
/*
  const Dtype* coeffs = bottom[1]->cpu_data();
  const Dtype* pixels0 = all_pixels0_.cpu_data();
  const Dtype* pixels1 = all_pixels1_.cpu_data();
  std::cout << "coeffs1\n";
  for (int i = 0; i < 64; ++i) {
    std::cout << std::setprecision(5) << coeffs[i];
    int j = i % 8;
    if (j == 7) {
      std::cout << ";\n";
    }
    else {
      std::cout << ", ";
    }
  }
  std::cout << "\n Pixels0:\n";
  for (int i = 0; i < 64; ++i) {
    std::cout << std::setprecision(5) << pixels0[i];
    int j = i % 8;
    if (j == 7) {
      std::cout << ";\n";
    }   
    else {
      std::cout << ", ";
    }   
  }
  std::cout << "\n";
  std::cout << "\n Pixels1:\n";
  for (int i = 0; i < 64; ++i) {
    std::cout << std::setprecision(5) << pixels1[i];
    int j = i % 8;
    if (j == 7) {
      std::cout << ";\n";
    }
    else {
      std::cout << ", ";
    }
  }
  std::cout << "\n";
*/
  caffe_gpu_sub(
     count,
     all_pixels0,
     all_pixels1,
     diff_.mutable_gpu_data());
/*
  const Dtype* diff = diff_.cpu_data();
  for (int i = 0; i < 64; ++i) {
    std::cout << std::setprecision(5) << diff[i];
    int j = i % 8;
    if (j == 7) {
      std::cout << ";\n";
    }
    else {
      std::cout << ", ";
    }
  }
  std::cout << std::endl;
*/
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* idct_derivs = idct2_derivs_.gpu_data();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      const int num_blocks = 9;
      const int block_size = 64;
      const int example_size = num_blocks * block_size;
      const int N = bottom[i]->count() / example_size;
      for (int j = 0; j < N; ++j) {
        // compute dE/dI * dI/dy for each example in the batch
        for (int k = 0; k < num_blocks; ++k) {
          const Dtype* diff = diff_.gpu_data() + j*example_size + k*block_size;
          Dtype* result = bottom[i]->mutable_gpu_diff() + j*example_size + k*block_size;
          // caffe_gpu_gemm assumes matrices are in column-major order,
          // so they need to be transposed
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, 64, 64,
                                alpha, diff, idct_derivs, 0., result);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(IdctEuclideanLossLayer);

}  // namespace caffe
