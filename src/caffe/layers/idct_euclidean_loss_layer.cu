#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
/*
template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  all_pixels0_.ReshapeLike(*bottom[0]);
  all_pixels1_.ReshapeLike(*bottom[0]);
  idct2_derivs_.ReshapeLike(*bottom[0]);
  caffe_gpu_get_didct2(1, idct2_derivs_.mutable_gpu_data());
  LOG(FATAL) << "CALLED DIDCT";
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  //all_pixels0_.ReshapeLike(*bottom[0]);
  //all_pixels1_.ReshapeLike(*bottom[0]);
  //idct2_derivs_.ReshapeLike(*bottom[0]);
}
*/
template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  const int block_size = 64;
  const int example_size = block_size * 9;

  const Dtype* c_coeffs0 = bottom[0]->gpu_data();
  const Dtype* c_coeffs1 = bottom[1]->gpu_data();

  // compute the 2d idct
  Dtype* all_pixels0 = all_pixels0_.mutable_gpu_data();
  Dtype* all_pixels1 = all_pixels1_.mutable_gpu_data();

  const int num_examples = count / example_size;
  caffe_gpu_idct2(num_examples, c_coeffs0, c_coeffs1, all_pixels0, all_pixels1);

  caffe_gpu_sub(
     count,
     all_pixels0,
     all_pixels1,
     diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* idct_derivs = idct2_derivs_.gpu_data();
/*  const Dtype* ptr = idct2_derivs_.cpu_data();
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      std::cout << ptr[i*64+j];
      if (j == 63) {
        std::cout << ";";
      }
      else {
        std::cout << ",";
      }
    }
  }
*/
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1; 
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      const int block_size = 64; 
      const int example_size = 9 * block_size;
      const int N = bottom[i]->count() / example_size;
      for (int j = 0; j < N; ++j) {
        // compute dE/dI * dI/dy for each example in the batch
        for (int k = 0; k < 9; ++k) {
          const Dtype* diff = diff_.gpu_data() + j*example_size + k*block_size;
          Dtype* result = bottom[i]->mutable_gpu_diff() + j*example_size + k*block_size;
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, 64, 64, 
                                alpha, diff, idct_derivs, 0., result);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(IdctEuclideanLossLayer);

}  // namespace caffe
