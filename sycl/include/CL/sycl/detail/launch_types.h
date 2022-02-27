//==------- launch_types.h - Describes the launch type of a kernel ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

enum class launch {
  max_occupancy,
  cooperative,
  none,
};
}
} // __SYCL_INLINE_NAMESPACE(cl)