#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <climits>
#include <cmath>  // for std::fabs
#include <cstdlib>  // for rand_r

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MathFunctionsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MathFunctionsTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  // http://en.wikipedia.org/wiki/Hamming_distance
  int ReferenceHammingDistance(const int n, const Dtype* x, const Dtype* y) {
    int dist = 0;
    uint64_t val;
    for (int i = 0; i < n; ++i) {
      if (sizeof(Dtype) == 8) {
        val = static_cast<uint64_t>(x[i]) ^ static_cast<uint64_t>(y[i]);
      } else if (sizeof(Dtype) == 4) {
        val = static_cast<uint32_t>(x[i]) ^ static_cast<uint32_t>(y[i]);
      } else {
        LOG(FATAL) << "Unrecognized Dtype size: " << sizeof(Dtype);
      }
      // Count the number of set bits
      while (val) {
        ++dist;
        val &= val - 1;
      }
    }
    return dist;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

template <typename Dtype>
class CPUMathFunctionsTest
  : public MathFunctionsTest<CPUDevice<Dtype> > {
};

TYPED_TEST_CASE(CPUMathFunctionsTest, TestDtypes);

TYPED_TEST(CPUMathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

TYPED_TEST(CPUMathFunctionsTest, TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  EXPECT_EQ(this->ReferenceHammingDistance(n, x, y),
            caffe_cpu_hamming_distance<TypeParam>(n, x, y));
}

TYPED_TEST(CPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam cpu_asum = caffe_cpu_asum<TypeParam>(n, x);
  EXPECT_LT((cpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(CPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sign<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sgnbit<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_cpu_scale<TypeParam>(n, alpha, this->blob_bottom_->cpu_data(),
                             this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestCopy) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  TypeParam* top_data = this->blob_top_->mutable_cpu_data();
  caffe_copy(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

// TODO: Fix caffe_gpu_hamming_distance and re-enable this test.
TYPED_TEST(GPUMathFunctionsTest, DISABLED_TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  x = this->blob_bottom_->gpu_data();
  y = this->blob_top_->gpu_data();
  int computed_distance = caffe_gpu_hamming_distance<TypeParam>(n, x, y);
  EXPECT_EQ(reference_distance, computed_distance);
}

TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam gpu_asum;
  caffe_gpu_asum<TypeParam>(n, this->blob_bottom_->gpu_data(), &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sign<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sgnbit<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_gpu_scale<TypeParam>(n, alpha, this->blob_bottom_->gpu_data(),
                             this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestCopy) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  caffe_copy(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(GPUMathFunctionsTest, testdIdy) {
  bool success = true;

  Blob<TypeParam> y_;
  y_.Reshape(1, 1, 64, 64);
  TypeParam* y = y_.mutable_cpu_data();

  Blob<TypeParam> idct2_y_;
  idct2_y_.Reshape(1, 1, 64, 64);
  TypeParam* idct2_y = idct2_y_.mutable_gpu_data();

  Blob<TypeParam> dIdy_;
  dIdy_.Reshape(1, 1, 64, 64);
  TypeParam* dIdy = dIdy_.mutable_gpu_data();

  TypeParam num_deriv;

  TypeParam y_orig[64] = {3887.29,-1131.98,1415.07,-107.701,897.421,229.441,642.426,444.358,-1131.98,329.635,-412.069,31.3625,-261.331,-66.8145,-187.076,-129.398,1415.07,-412.069,515.118,-39.206,326.683,83.523,233.857,161.758,-107.701,31.3625,-39.206,2.9845,-24.864,-6.3565,-17.8,-12.3105,897.421,-261.331,326.683,-24.864,207.179,52.97,148.31,102.585,229.441,-66.8145,83.523,-6.3565,52.97,13.5435,37.918,26.2275,642.426,-187.076,233.858,-17.8,148.31,37.918,106.168,73.436,444.358,-129.398,161.758,-12.3105,102.585,26.2275,73.436,50.7945,};

  static const int s_zigzag_lookup[8][8] = {
    { 0, 1, 5, 6, 14, 15, 27, 28 },
    { 2, 4, 7, 13, 16, 26, 29, 42 },
    { 3, 8, 12, 17, 25, 30, 41, 43 },
    { 9, 11, 18, 24, 31, 40, 44, 53 },
    { 10, 19, 23, 32, 39, 45, 52, 54 },
    { 20, 22, 33, 38, 46, 51, 55, 60 },
    { 21, 34, 37, 47, 50, 56, 59, 61 },
    { 35, 36, 48, 49, 57, 58, 62, 63 }
  };

  // put y in zigzag order
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      y[i * 8 + j] = y_orig[s_zigzag_lookup[i][j]];
    }
  }

  caffe_gpu_idct2(1, y_.gpu_data(), idct2_y);
  caffe_gpu_get_didct2(1, dIdy);

  TypeParam eps_val = (TypeParam)1e-5;

  // how much idct i changes when you change coeff j
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      // add eps to coeff j
      Blob<TypeParam> eps_;
      eps_.Reshape(1, 1, 64, 64);
      TypeParam* eps = eps_.mutable_cpu_data();
      memset(eps, 0, 64*sizeof(*eps));
      eps[j] = eps_val;

      Blob<TypeParam> y_eps_;
      y_eps_.Reshape(1, 1, 64, 64);
      TypeParam* y_eps = y_eps_.mutable_cpu_data();

      for (int k = 0; k < 64; ++k) {
        y_eps[k] = y_orig[k] + eps[k];
      }

      TypeParam y2_eps[64];
      // put y1 in zigzag order
      for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
          y2_eps[a * 8 + b] = y_eps[s_zigzag_lookup[a][b]];
        }
      }
      for (int a = 0; a < 64; ++a) y_eps[a] = y2_eps[a];

      Blob<TypeParam> idct2_y_eps_;
      idct2_y_eps_.Reshape(1, 1, 64, 64);
      TypeParam* idct2_y_eps = idct2_y_eps_.mutable_gpu_data();

      caffe_gpu_idct2(1, y_eps_.gpu_data(), idct2_y_eps);

      num_deriv = (idct2_y_eps_.cpu_data()[i] - idct2_y_.cpu_data()[i]) / eps_val;

      TypeParam diff = fabs(num_deriv - dIdy_.cpu_data()[i*64+j]);
      success &= (diff < eps_val);

    }
  }

  EXPECT_EQ(success, true);
}

TYPED_TEST(GPUMathFunctionsTest, testDIdy_z)
{
  bool success = true;
  // how much idct2 changes when you change coefficients

  // coefficients
  TypeParam y_orig[64] = {3887.29,-1131.98,229.441,642.426,-187.076,-129.398,2.9845,-24.864,1415.07,897.421,444.358,-66.8145,1415.07,-39.206,-6.3565,83.523,-107.701,-1131.98,-261.331,-412.069,31.3625,-17.8,-66.8145,-6.3565,329.635,31.3625,515.118,-107.701,-12.3105,229.441,52.97,37.918,-412.069,-39.206,161.758,897.421,102.585,13.5435,148.31,106.168,326.683,233.857,-261.331,148.31,37.918,-17.8,73.436,102.585,83.523,326.683,52.97,26.2275,233.858,444.358,-12.3105,26.2275,-24.864,207.179,642.426,-187.076,-129.398,161.758,73.436,50.7945,};

  Blob<TypeParam> dIdy_;
  dIdy_.Reshape(1, 1, 64, 64);
  TypeParam* dIdy = dIdy_.mutable_gpu_data();
  caffe_gpu_get_didct2(1, dIdy);

  Blob<TypeParam> y_;
  y_.Reshape(1, 1, 64, 64);
  TypeParam* y = y_.mutable_cpu_data();
  for (int i = 0; i < 64; ++i) {
    y[i] = y_orig[i];
  }
  Blob<TypeParam> idct2_y_;
  idct2_y_.Reshape(1, 1, 64, 64);
  caffe_gpu_idct2(1, y_.gpu_data(), idct2_y_.mutable_gpu_data());

  // now get numerical derivative (how much idct changes when you change
  // the jth coeff
  const TypeParam eps_val = (TypeParam)1e-4f;
  for (int i = 0; i < 64; ++i) { // loop through idct
    for (int j = 0; j < 64; ++j) { // loop through coeffs
      // add eps to coeff
      Blob<TypeParam> eps_;
      eps_.Reshape(1, 1, 64, 64);
      TypeParam* eps = eps_.mutable_cpu_data();
      memset(eps, 0, sizeof(*eps));
      eps[j] = eps_val;

      Blob<TypeParam> y_eps_;
      y_eps_.Reshape(1, 1, 64, 64);
      TypeParam* y_eps = y_eps_.mutable_cpu_data();
      for (int k = 0; k < 64; ++k) {
        y_eps[k] = y[k] + eps[k];
      }

      Blob<TypeParam> idct2_y_eps_;
      idct2_y_eps_.Reshape(1, 1, 64, 64);
      caffe_gpu_idct2(1, y_eps_.gpu_data(), idct2_y_eps_.mutable_gpu_data());

      TypeParam num_deriv = (idct2_y_eps_.cpu_data()[i] - idct2_y_.cpu_data()[i]) / eps_val;
      success &= fabs(num_deriv - dIdy_.cpu_data()[i*64+j]) < eps_val;
    }
  }
  EXPECT_EQ(success, true);
}

TYPED_TEST(GPUMathFunctionsTest, testIdctDerivFromCoeffs) {
  bool success = true;
  const int N = 64;

  Blob<TypeParam> y_, y_bar_, dIdy_;
  y_.Reshape(1, 1, 64, 64);
  y_bar_.Reshape(1, 1, 64, 64);
  dIdy_.Reshape(1, 1, 64, 64);
  Blob<TypeParam> idct2_y_, idct2_y_bar_;
  idct2_y_.Reshape(1, 1, 64, 64);
  idct2_y_bar_.Reshape(1, 1, 64, 64);
  Blob<TypeParam> dEdy_, idct2_diff_;
  dEdy_.Reshape(1, 1, 64, 64);
  idct2_diff_.Reshape(1, 1, 64, 64);

  TypeParam* y = y_.mutable_cpu_data();
  TypeParam* y_bar = y_bar_.mutable_cpu_data();
  TypeParam* dIdy = dIdy_.mutable_gpu_data();
  TypeParam* idct2_y = idct2_y_.mutable_gpu_data();
  TypeParam* idct2_y_bar = idct2_y_bar_.mutable_gpu_data();
  TypeParam* dEdy = dEdy_.mutable_gpu_data();

  // coefficients in zigzag order
  TypeParam y_orig[64] = {3887.29,-1131.98,229.441,642.426,-187.076,-129.398,2.9845,-24.864,1415.07,897.421,444.358,-66.8145,1415.07,-39.206,-6.3565,83.523,-107.701,-1131.98,-261.331,-412.069,31.3625,-17.8,-66.8145,-6.3565,329.635,31.3625,515.118,-107.701,-12.3105,229.441,52.97,37.918,-412.069,-39.206,161.758,897.421,102.585,13.5435,148.31,106.168,326.683,233.857,-261.331,148.31,37.918,-17.8,73.436,102.585,83.523,326.683,52.97,26.2275,233.858,444.358,-12.3105,26.2275,-24.864,207.179,642.426,-187.076,-129.398,161.758,73.436,50.7945,};

  for(int i = 0; i < 64; ++i) {
    y[i] = y_orig[i];
    y_bar[i] = y_orig[i];
  }
  y_bar[0] += (TypeParam)1.0;
  y_bar[37] += (TypeParam)1.0;

  caffe_gpu_get_didct2(1, dIdy);

  TypeParam eps_val = (TypeParam)1e-4;
  caffe_gpu_idct2(1, y_.mutable_gpu_data(), idct2_y);
  caffe_gpu_idct2(1, y_bar_.mutable_gpu_data(), idct2_y_bar);

  caffe_gpu_sub(
     N,
     idct2_y_bar_.gpu_data(),
     idct2_y_.gpu_data(),
     idct2_diff_.mutable_gpu_data());
  caffe_gpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 1, 64, 64,
                        (TypeParam)(1.0/N), idct2_diff_.gpu_data(), dIdy_.gpu_data(), 0., dEdy);

  // compare with numerical derivative
  for (int i = 0; i < 64; ++i) {
    Blob<TypeParam> eps_;
    eps_.Reshape(1, 1, 64, 64);
    TypeParam* eps = eps_.mutable_cpu_data();
    memset(eps, 0, 64*sizeof(*eps));
    eps[i] = eps_val;

    Blob<TypeParam> y_bar_eps_;
    y_bar_eps_.Reshape(1, 1, 64, 64);
    TypeParam* y_bar_eps = y_bar_eps_.mutable_cpu_data();
    for (int k = 0; k < 64; ++k) {
      y_bar_eps[k] = y_bar[k] + eps[k];
    }

    Blob<TypeParam> idct2_y_bar_eps_;
    idct2_y_bar_eps_.Reshape(1, 1, 64, 64);
    caffe_gpu_idct2(1, y_bar_eps_.gpu_data(), idct2_y_bar_eps_.mutable_gpu_data());

    TypeParam E1 = (TypeParam)0.0, E2 = (TypeParam)0.0;
    for (int k = 0; k < 64; ++k) {
      E1 += (idct2_y_bar_.cpu_data()[k] - idct2_y_.cpu_data()[k]) * (idct2_y_bar_.cpu_data()[k]- idct2_y_.cpu_data()[k]);
      E2 += (idct2_y_bar_eps_.cpu_data()[k] - idct2_y_.cpu_data()[k]) * (idct2_y_bar_eps_.cpu_data()[k] - idct2_y_.cpu_data()[k]);
    }
    E1 = (TypeParam)(E1 / 128.0);
    E2 = (TypeParam)(E2 / 128.0);

    TypeParam num_deriv = (E2 - E1) / eps_val;
    success = success && fabs(dEdy_.cpu_data()[i] - num_deriv) < eps_val;

  }

  EXPECT_EQ(success, true);
}

#endif


}  // namespace caffe
