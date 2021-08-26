//==-------- array.hpp --- SYCL common iteration object --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/exception.hpp>
#include <functional>
#include <stdexcept>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
template <int dimensions> class id;
template <int dimensions> class range;
namespace detail {

template <int dimensions = 1> class array;

template <> class array<1> {

public:
  /* The following constructor is only available in the array struct
   * specialization where: dimensions==1 */
  array(size_t dim0 = 0) : v0{dim0} {}

  // Conversion operators to derived classes
  operator cl::sycl::id<1>() const {
    cl::sycl::id<1> result;
    result.v0 = v0;
    return result;
  }

  operator cl::sycl::range<1>() const {
    cl::sycl::range<1> result;
    result.v0 = v0;
    return result;
  }

  size_t get(int dimension) const {
    check_dimension(dimension);
    return v0;
  }

  size_t &operator[](int dimension) {
    check_dimension(dimension);
    return v0;
  }

  size_t operator[](int dimension) const {
    check_dimension(dimension);
    return v0;
  }

  array(const array<1> &rhs) = default;
  array(array<1> &&rhs) = default;
  array<1> &operator=(const array<1> &rhs) = default;
  array<1> &operator=(array<1> &&rhs) = default;

  // Returns true iff all elements in 'this' are equal to
  // the corresponding elements in 'rhs'.
  bool operator==(const array<1> &rhs) const { return this->v0 == rhs.v0; }

  // Returns true iff there is at least one element in 'this'
  // which is not equal to the corresponding element in 'rhs'.
  bool operator!=(const array<1> &rhs) const { return this->v0 != rhs.v0; }

protected:
  size_t v0;
  __SYCL_ALWAYS_INLINE void check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension != 1) {
      throw cl::sycl::invalid_parameter_error("Index out of range",
                                              PI_INVALID_VALUE);
    }
#endif
    (void)dimension;
  }
};

template <> class array<2> {

public:
  /* The following constructors are only available in the array struct
   * specialization where: dimensions==2 */
  array(size_t dim0, size_t dim1) : v0{dim0}, v1{dim1} {}

  array() : array(0, 0) {}

  // Conversion operators to derived classes
  operator cl::sycl::id<2>() const {
    cl::sycl::id<2> result;
    result.v0 = v0;
    result.v1 = v1;
    return result;
  }

  operator cl::sycl::range<2>() const {
    cl::sycl::range<2> result;
    result.v0 = v0;
    result.v1 = v1;
    return result;
  }

  size_t get(int dimension) const {
    check_dimension(dimension);
    return dimension == 0 ? v0 : v1;
  }

  size_t &operator[](int dimension) {
    check_dimension(dimension);
    return dimension == 0 ? v0 : v1;
  }

  size_t operator[](int dimension) const {
    check_dimension(dimension);
    return dimension == 0 ? v0 : v1;
  }

  array(const array<2> &rhs) = default;
  array(array<2> &&rhs) = default;
  array<2> &operator=(const array<2> &rhs) = default;
  array<2> &operator=(array<2> &&rhs) = default;

  // Returns true iff all elements in 'this' are equal to
  // the corresponding elements in 'rhs'.
  bool operator==(const array<2> &rhs) const {
    return v0 == rhs.v0 && v1 == rhs.v1;
  }

  // Returns true iff there is at least one element in 'this'
  // which is not equal to the corresponding element in 'rhs'.
  bool operator!=(const array<2> &rhs) const {
    return v0 != rhs.v0 || v1 != rhs.v1;
  }

protected:
  size_t v0, v1;
  __SYCL_ALWAYS_INLINE void check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension != 2) {
      throw cl::sycl::invalid_parameter_error("Index out of range",
                                              PI_INVALID_VALUE);
    }
#endif
    (void)dimension;
  }
};

template<> class array<3> {

public:
  array(size_t dim0, size_t dim1, size_t dim2) : v0{dim0}, v1{dim1}, v2{dim2} {}

  array() : array(0, 0, 0) {}

  // Conversion operators to derived classes
  operator cl::sycl::id<3>() const {
    cl::sycl::id<3> result;
    result.v0 = v0;
    result.v1 = v1;
    result.v2 = v2;
    return result;
  }

  operator cl::sycl::range<3>() const {
    cl::sycl::range<3> result;
    result.v0 = v0;
    result.v1 = v1;
    result.v2 = v2;
    return result;
  }

  size_t get(int dimension) const {
    check_dimension(dimension);
    return dimension == 0 ? v0 : (dimension == 1 ? v1 : v2);
  }

  size_t &operator[](int dimension) {
    check_dimension(dimension);
    return dimension == 0 ? v0 : (dimension == 1 ? v1 : v2);
  }

  size_t operator[](int dimension) const {
    check_dimension(dimension);
    return dimension == 0 ? v0 : (dimension == 1 ? v1 : v2);
  }

  array(const array<3> &rhs) = default;
  array(array<3> &&rhs) = default;
  array<3> &operator=(const array<3> &rhs) = default;
  array<3> &operator=(array<3> &&rhs) = default;

  // Returns true iff all elements in 'this' are equal to
  // the corresponding elements in 'rhs'.
  bool operator==(const array<3> &rhs) const {
    return v0 == rhs.v0 && v1 == rhs.v1 && v2 == rhs.v2;
  }

  // Returns true iff there is at least one element in 'this'
  // which is not equal to the corresponding element in 'rhs'.
  bool operator!=(const array<3> &rhs) const {
    return v0 != rhs.v0 || v1 != rhs.v1 || v2 != rhs.v2;
  }

protected:
  size_t v0, v1, v2;
  __SYCL_ALWAYS_INLINE void check_dimension(int dimension) const {
#ifndef __SYCL_DEVICE_ONLY__
    if (dimension != 3) {
      throw cl::sycl::invalid_parameter_error("Index out of range",
                                              PI_INVALID_VALUE);
    }
#endif
    (void)dimension;
  }
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
