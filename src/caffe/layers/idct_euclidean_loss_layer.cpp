#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::computeDIdy(Dtype derivs[4096])
{
  for (int m = 0; m < 8; ++m) {
    for (int n = 0; n < 8; ++n) {
      int i = m * 8 + n;
      for (int p = 0; p < 8; ++p) {
        for (int q = 0; q < 8; ++q) {
          int j = p * 8 + q;
          Dtype a_p = (Dtype)(sqrt(2.0 / 8.0));
          Dtype a_q = (Dtype)(sqrt(2.0 / 8.0));
          if (p == 0) {
            a_p = (Dtype)(1.0 / sqrt(8.0));
          }
          if (q == 0) {
            a_q = (Dtype)(1.0 / sqrt(8.0));
          }
          derivs[64*i + j] = (Dtype)(a_p * a_q * cos(M_PI*(2*m+1)*p/16.0) * cos(M_PI*(2*n+1)*q/16.0));
        }
      }
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::getCenter12x12(Dtype* coeffs)
{
  const int block_size = 64;
  // resize coeffs from array of size 1x576 to 24x24 image
  Dtype block24_24[24][24];
  for (int i = 0; i < 9; ++i) {
    Dtype* block = coeffs + i*block_size; // current 8x8 block
    int j = i/3;
    int k = i%3; // indices into 3x3 block of 64 coeffs each
    for (int r = 0; r < 8; ++r) { // go through each coeff in the current 8x8 block
      for (int c = 0; c < 8; ++c) {
        block24_24[8*j+r][8*k+c] = block[r*8+c];
      }
    }
  }

  // take out the middle 12x12
  for (int j = 0; j < 24; ++j) {
    for (int i = 0; i < 24; ++i) {
      if ( !(j >= 6 && j <= 17 && i >=6 && i <= 17) ) {
        block24_24[j][i] = 0.0;
      }
    }
  }

  // write the thing back to a 1x576 array, block by block
  Dtype* pixels = coeffs;
  for (int i = 0; i < 9; ++i) {
    int j = i/3;
    int k = i%3;
    for (int r = 0; r < 8; ++r) { // go through each coeff in the current 8x8 block
      for (int c = 0; c < 8; ++c) {
        *pixels = block24_24[8*j+r][8*k+c];
        pixels++;
      }
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::computeIdct2(const Dtype* c_coeffs, Dtype* local_pixels)
{
  // compute the 2d idct
  for (int i = 0; i < 64; ++i)
  {
    int m = i / 8;
    int n = i % 8;
    local_pixels[i] = 0.0;
    for (int p = 0; p < 8; ++p)
    {
      for (int q = 0; q < 8; ++q)
      {
        Dtype a_p = (Dtype)sqrt(2.0 / 8.0);
        Dtype a_q = (Dtype)sqrt(2.0 / 8.0);
        if (p == 0) {
          a_p = (Dtype)(1.0 / sqrt(8));
        }
        if (q == 0) {
          a_q = (Dtype)(1.0 / sqrt(8));
        }
        local_pixels[i] += (Dtype)(a_p * a_q * c_coeffs[p*8+q] * cos(M_PI*p*(2*m+1)/16) * cos(M_PI*q*(2*n+1)/16));
      }
    }
  }
}

template <typename Dtype>
void IdctEuclideanLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  zigzag_indices_[0][0] = 0;
  zigzag_indices_[0][1] = 1;
  zigzag_indices_[0][2] = 5;
  zigzag_indices_[0][3] = 6;
  zigzag_indices_[0][4] = 14;
  zigzag_indices_[0][5] = 15;
  zigzag_indices_[0][6] = 27;
  zigzag_indices_[0][7] = 28;

  zigzag_indices_[1][0] = 2;
  zigzag_indices_[1][1] = 4;
  zigzag_indices_[1][2] = 7;
  zigzag_indices_[1][3] = 13;
  zigzag_indices_[1][4] = 16;
  zigzag_indices_[1][5] = 26;
  zigzag_indices_[1][6] = 29;
  zigzag_indices_[1][7] = 42;

  zigzag_indices_[2][0] = 3;
  zigzag_indices_[2][1] = 8;
  zigzag_indices_[2][2] = 12;
  zigzag_indices_[2][3] = 17;
  zigzag_indices_[2][4] = 25;
  zigzag_indices_[2][5] = 30;
  zigzag_indices_[2][6] = 41;
  zigzag_indices_[2][7] = 43;

  zigzag_indices_[3][0] = 9;
  zigzag_indices_[3][1] = 11;
  zigzag_indices_[3][2] = 18;
  zigzag_indices_[3][3] = 24;
  zigzag_indices_[3][4] = 31;
  zigzag_indices_[3][5] = 40;
  zigzag_indices_[3][6] = 44;
  zigzag_indices_[3][7] = 53;

  zigzag_indices_[4][0] = 10;
  zigzag_indices_[4][1] = 19;
  zigzag_indices_[4][2] = 23;
  zigzag_indices_[4][3] = 32;
  zigzag_indices_[4][4] = 39;
  zigzag_indices_[4][5] = 45;
  zigzag_indices_[4][6] = 52;
  zigzag_indices_[4][7] = 54;

  zigzag_indices_[5][0] = 20;
  zigzag_indices_[5][1] = 22;
  zigzag_indices_[5][2] = 33;
  zigzag_indices_[5][3] = 38;
  zigzag_indices_[5][4] = 46;
  zigzag_indices_[5][5] = 51;
  zigzag_indices_[5][6] = 55;
  zigzag_indices_[5][7] = 60;

  zigzag_indices_[6][0] = 21;
  zigzag_indices_[6][1] = 34;
  zigzag_indices_[6][2] = 37;
  zigzag_indices_[6][3] = 47;
  zigzag_indices_[6][4] = 50;
  zigzag_indices_[6][5] = 56;
  zigzag_indices_[6][6] = 59;
  zigzag_indices_[6][7] = 61;

  zigzag_indices_[7][0] = 35;
  zigzag_indices_[7][1] = 36;
  zigzag_indices_[7][2] = 48;
  zigzag_indices_[7][3] = 49;
  zigzag_indices_[7][4] = 57;
  zigzag_indices_[7][5] = 58;
  zigzag_indices_[7][6] = 62;
  zigzag_indices_[7][7] = 63;

  computeDIdy(idct_derivs_);
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
