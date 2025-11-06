#ifndef DEF_CUH
#define DEF_CUH


#include <stdlib.h>
#include <stdio.h> /* IWYU pragma: keep since stderr needs this */
#include <cstdint>

#ifndef __CUDACC__ /* nvcc with c++?? */
#error "this is a cuda header!"
#endif

template <typename A, typename B>
struct is_same {
  static constexpr bool value = false;
};

template <typename A>
struct is_same<A, A> {
  static constexpr bool value = true;
};

template<bool b, typename T, typename F>
struct _cond_type {
  using type = F;
};

template<typename T, typename F>
struct _cond_type<true, T, F> {
  using type = T;
};

template<bool b, typename T, typename F>
using cond_type = typename _cond_type<b, T, F>::type;

#define SAME_TYPE(X,Y) is_same<X,Y>::value
#define COND_TYPE(B,T,F) cond_type<B,T,F>

template <std::size_t N> struct _textype_for_size;

template <> struct _textype_for_size<1> { using type = uint8_t;  };
template <> struct _textype_for_size<2> { using type = uint16_t; };
template <> struct _textype_for_size<4> { using type = uint32_t; };
template <> struct _textype_for_size<8> { using type = uint2; };

template <std::size_t N>
struct __textype_for_size {
    static_assert(N == 1 || N == 2 || N == 4 || N == 8,
                  "No standard uint type available for this size");
    using type = typename _textype_for_size<N>::type;
};

#define AS_TEXTYPE(T) typename __textype_for_size<sizeof(T)>::type

template <typename T>
__host__ __forceinline__ auto cuda_create_channel_desc_custom() {
        return cudaCreateChannelDesc<AS_TEXTYPE(T)>();
}

template <typename T>
__device__ __forceinline__ T tex_1d_fetch_custom(cudaTextureObject_t texObj, int id) {
        auto src = tex1Dfetch<AS_TEXTYPE(T)>(texObj, id);
        return *reinterpret_cast<T*>(&src);
}

#endif