#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

std::ostream& operator<<(std::ostream& out, __m128i v) {
  int* vals = (int*)(&v);
  out << vals[0] << ' ' << vals[1] << ' ' << vals[2] << ' ' << vals[3];
  return out;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
  out << "[";
  if (!vec.empty()) {
    out << vec.front();
  }
  for (unsigned i = 1, sz = vec.size(); i < sz; i++) {
    out << ", " << vec[i];
  }
  out << "]";
  return out;
}

// bool simd_search(int* array, int n, int v) {
//   if (array[n - 1] <= v) {
//     return array[n - 1] == v;
//   }
//   --n; // now we need to find v with array[0, n-1) while accessing array[n] is useless but valid
//   static __m128i shifts = _mm_set_epi32(4, 3, 2, 1);
//   __m128i xv = _mm_set1_epi32(v);
//   int l = 0;
//   while (n - l >= 3) {
//     int s = l + n;
//     // (l+r), by r I mean n
//     __m128i indices = _mm_set1_epi32(s);
//     simd_print(indices);
//     // (l+r)*[1..4], need sse4_1, use `cat /proc/cpuinfo` to check.
//     // note that this may exceed the range of `int`
//     indices = _mm_mullo_epi32(indices, shifts);
//     simd_print(indices);
//     // (l+r)*[1..4] >> 2
//     indices = _mm_srli_epi32(indices, 2);
//     std::cout<<"=>";simd_print(indices);
//     // array[(l+r)*[1..4] >> 2]
//     __m128i values = _mm_i32gather_epi32(array, indices, 4);
//     simd_print(values);
//     // compare array[(l+r)*[1..4] >> 2] with value
//     __m128i cmplt = _mm_cmplt_epi32(values, xv);
//     simd_print(cmplt);
//     __m128i cmpeq = _mm_cmpeq_epi32(values, xv);
//     simd_print(cmpeq);
//     // 1110, 1100, 1000, 0000 => [indices[2], [3]==n), [[1], [2]), [[0], [1]), [l, [0])
//     __m128i cmple = _mm_or_si128(cmplt, cmpeq);
//     simd_print(cmple);
//     return false;
//   }
// }

bool simd_search2(int* array, int n, int v) {
  static const __m128i ones = _mm_set1_epi32(1);
  static const __m128i mul_shifts = _mm_set_epi32(4, 3, 2, 1);
  if (n >= 5) {
    __m128i xv = _mm_set1_epi32(v);
    __m128i indices, values, cmpgt;
    int* cmp = (int*)(&cmpgt);
    int* idx = (int*)(&indices);
    // n is the len of the array
    // array will be updated to the latest head of the search range
    for (int s = n / 5; n > 5; s = n / 5) {
      indices = _mm_set1_epi32(s);
      indices = _mm_mullo_epi32(indices, mul_shifts);
      values = _mm_i32gather_epi32(array, indices, 4);
      cmpgt = _mm_and_si128(_mm_cmplt_epi32(xv, values), ones);
      // v < value ? left : right
      int i = 4 - (cmp[0] + cmp[1] + cmp[2] + cmp[3]);
      if (i > 0) {
        array += idx[i - 1];
        n = (i == 4 ? n - idx[3] : idx[i] - idx[i - 1]);
      } else {
        n = idx[0];
      }
      assert(n >= 0);
    }
  }
  return std::find(array, array + n, v) != array + n;
}

template<typename T>
struct KaryTreeLinearizer {
  std::vector<T> linearize(const std::vector<T>& sorted) {
    std::vector<T> tree(sorted.size());
    cur_sorted = &sorted;
    cur_tree = &tree;
    sorted_idx = 0;
    num_elements = sorted.size();
    num_blocks = (num_elements + 3) >> 2;
    inner_linearize();
    return tree;
  }

private:
  const std::vector<T>* cur_sorted;
  std::vector<T>* cur_tree;
  unsigned sorted_idx;
  unsigned num_blocks;
  unsigned num_elements;

  void inner_linearize(unsigned blk = 0) {
    if (blk >= num_blocks) {
      return;
    }
    auto tidx = blk << 2;
    blk += blk << 2 | 1;
    inner_linearize(blk);
    for (unsigned i = 0; i < 4; i++) {
      if (tidx >= num_elements) {
        return;
      }
      cur_tree->operator[](tidx++) = cur_sorted->operator[](sorted_idx++);
      inner_linearize(++blk);
    }
  }
};

#define _mm_cmple_epu32(a, b) _mm_cmpeq_epi32(_mm_min_epu32(a, b), a)

bool kary_tree_search(int* tree, int n, int v) {
  static const __m128i ones = _mm_set1_epi32(1);
  int blk_idx = 0;
  __m128i vals = _mm_set1_epi32(v);
  __m128i cmple;
  int* cmples = (int*)(&cmple);
  for (auto num_blks = (n + 3) >> 2; blk_idx < num_blks;) {
    __m128i blk = *(__m128i*)(tree + (blk_idx << 2));
    // std::cout << blk_idx << ' ' << blk << '[' << v << ']' << std::endl;
    int cmpeq = _mm_movemask_epi8(_mm_cmpeq_epi32(blk, vals));
    if (cmpeq) {
      return true;
    }
    cmple = _mm_and_si128(_mm_cmple_epu32(blk, vals), ones);
    // int adv = cmples[0] + cmples[1] + cmples[2] + cmples[3];
    cmple = _mm_hadd_epi32(cmple, cmple);
    cmple = _mm_hadd_epi32(cmple, cmple);
    int adv = cmples[0];
    // blk_idx = blk_idx * 5 + adv + 1;
    blk_idx += (blk_idx << 2 | 1) + adv;
  }
  return false;
}

std::chrono::milliseconds now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch());
}

void test(size_t sz, size_t n_search) {
  srand(time(0));
  std::vector<int> ints(sz);
  for (auto& i : ints) {
    i = rand() % ints.size();
  }
  std::sort(ints.begin(), ints.end());
  std::vector<int> nums(n_search);
  for (auto& i : nums) {
    i = rand() % ints.size();
  }
  if (sz < 10000) {
    int bfind = 0;
    auto st = now();
    for (auto& i : nums) {
      bfind += std::find(ints.begin(), ints.end(), i) != ints.end();
    }
    auto ed = now();
    std::cout << "std::find finds " << bfind << " times using " << (ed - st).count() << " ms." << std::endl;
  } else {
    std::cout << "std::find is skipped" << std::endl;
  }
  {
    int bfind = 0;
    auto st = now();
    for (auto& i : nums) {
      bfind += std::binary_search(ints.begin(), ints.end(), i);
    }
    auto ed = now();
    std::cout << "std::binary_search finds " << bfind << " times using " << (ed - st).count() << " ms." << std::endl;
  }
  {
    int sfind = 0;
    auto st = now();
    for (auto& i : nums) {
      sfind += simd_search2(&ints.front(), ints.size(), i);
    }
    auto ed = now();
    std::cout << "SIMD search finds " << sfind << " times using " << (ed - st).count() << " ms." << std::endl;
  }
  {
    int kfind = 0;
    auto st = now();
    KaryTreeLinearizer<int> linearizer;
    auto tree = linearizer.linearize(ints);
    auto md = now();
    for (auto& i : nums) {
      kfind += kary_tree_search(&tree.front(), ints.size(), i);
    }
    auto ed = now();
    std::cout << "Kary tree search finds " << kfind << " times using " << (md - st).count() << " ms for pre-processing and " << (ed - md).count() << " ms for searching." << std::endl;
  }
}

int main() {
  for (size_t i = 0, sz = 8; i < 25; i++, sz <<= 1) {
    std::cout << "Testing size: " << sz << std::endl;
    test(sz, 10000000);
  }
}
