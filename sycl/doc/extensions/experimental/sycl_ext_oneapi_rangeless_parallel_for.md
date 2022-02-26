# sycl_ext_oneapi_rangeless_parallel_for

## Notice

This document describes an **experimental** API that applications can use to try
out a new feature. Future versions of this API may change in ways that are
incompatible with this experimental version.

## Motivations

Reliably computing the best launch parameters for a `parallel_for` kernel and the approach depends strongly on the target hardware.
Actually SYCL provides no way to query a reduction kernel for the maximum work-group size allowed. A workaround is to launch kernels using a `range` instead of a `nd_range` and let the runtime use the best values. However this results in the loss of some of the sycl features related to work-items (barriers, local memory,...). A solution would be to let the runtime decide on the `nd_range` and let the kernel adapt to the launch parameters.

Some devices support _inter-work-group_ synchronizations (such as device-wide barriers). One of the main benefit of such synchronizations is to reduce the number of kernels an application submits. An implementation example would be **NVIDIA Cooperative Groups**. Using these features require a more strict control of the kernel execution range as well as using specific driver calls to submit the kernel. On NVIDIA GPUs, the kernel execution range depends on compile-time parameters such as register usage, static local memory usage, stack-frame as well as runtime parameters (dynamic local memory).

Some of these are not accessible in SYCL right now. Even if they were, the computation is highly device and implementation dependant.

## Introduction

This extension introduces **rangeless `parallel_for`** where the usual one dimensional execution range is replaced by a **`sycl::launch`** attributes that will let the runtime know how to launch the kernel.

This extension could also consist in introducing **`sycl::launch`** attributes first and as a corollary allowing **rangeless `parallel_for`** launches for some of these attributes.

## Feature test macro

`SYCL_EXT_ONEAPI_RANGELESS_PARALLEL_FOR` defined to 1 if the feature is available.

## New launch attributes

| Launch tag                    | Description                                                                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `sycl::launch::max_occupancy` | Implicitly computes the `nd_range<1>` that maximizes the occupancy of the kernel on the device and launches the kernel using that range. |
| `sycl::launch::cooperative`   | Launches a cooperative kernel where forward progress of all work-items is guaranteed.                                                    |

## Examples

First attempt to a rangeless-parallel_for:

```C++
// We do not provide a range to the kernel launch
q.parallel_for(sycl::launch::max_occupancy, [=](sycl::nd_item<1> it) {
    if(i.get_global_linear_id() >= momentums.size) return;
    momentums[i] *= alpha;
}).wait();
```

This attempt is obviously wrong as we might be processing to few elements in the `momentums` vector.

To use properly this extension the user would need to manually oversubscribe the work-items with "work-units". This can be easily abstracted with a `occupancy_range_adapter` (code at the end) to perform `size` "work-units" across the whole kernel range.

The code now becomes:

```C++
q.parallel_for(sycl::launch::max_occupancy, [=](sycl::nd_item<1> it) {
    // The work still has to be performed the requested work size, so we'll perform several 'work-units' per work-item.
    occupancy_range_adapter(size, it, [&](size_t i) {
        // The `occupancy_range_adapter` will take care of oversubscribing the work-item with 'work-units'.
        momentums[i] *= alpha;
    });
}).wait();
```

### Device-wide

The following example uses `sycl::launch::max_occupancy` and shows how to use device-wide synchronizations to center an array of float (so its barycenter falls on 0) in a single kernel launch.

If we want to perform a single reduction, we will have necessarily to synchronize the whole device/work-range. Using `sycl::launch::max_occupancy` gives us _forward progress guarantees_ meaning that we can assume that all the work-items from the `nd_item<1>` will be running at the same time. Once again this is achieved because we're decoupling the work size from the kernel work range.

```C++
/* SYCL Boilerplate for a parallel reduction */
using local_atomic_ref = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>;
using global_atomic_ref = sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
using counter_atomic_ref = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>;
auto global_reducer_b = sycl::buffer<float>(1);
auto barrier_counter_b = sycl::buffer<int>(1);

q.submit([&](sycl::handler& cgh) {
    auto local_reducer = sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>(1, cgh);
    auto global_reducer = sycl::accessor{global_reducer_b, cgh};
    auto barrier_counter = sycl::accessor{barrier_counter_b, cgh};

    /* Launching the rangeless parallel_for kernel */
    cgh.parallel_for(sycl::launch::cooperative, [=, size = size_, momentums = momentums_.get()](sycl::nd_item<1> item) {
        /* Computing the barycenter :  Local reduction */
        {
            auto thread_reducer = float{};
            /* per-thread reduction */
            occupancy_range_adapter(size, item, [&](size_t i) { thread_reducer += momentums[i]; });
            
            /* Work-group-wide reduction into local memory */
            local_atomic_ref{local_reducer[0]} += thread_reducer;
        }

        /* Waiting for all the work-items to do the reduction. */
        item.barrier();   
        
        /* Device-wide reduction */
        if (item.get_local_id(0) == 0) {
            /* Reduce the work-group result into global memory */
            global_atomic_ref{global_reducer[0]} += local_atomic_ref{local_reducer[0]};

            /* Register the "leader" of each work-group on the barrier */
            counter_atomic_ref(barrier_counter[0])++;
            /* Device-wide Sync using a spinlock */

            while (counter_atomic_ref(barrier_counter[0]).load() != item.get_group_range(0));
            /* Barried passed, we fetch the result, compute the average and store it into local memory */
            local_atomic_ref{local_reducer[0]} = global_atomic_ref{global_reducer[0]} / static_cast<float>(size);      
        }
        
        /* Everyone waits untill average is available in local_reducer */
        item.barrier();
        /* Finally, we use the average to perform the centering */
        occupancy_range_adapter(size, item, [&](size_t i) { momentums[i] -= local_reducer[0]; });
    });
}).wait();
```

We could have asked the user to get a `nd_range` through a query, but because this feature could required a specific launch in the driver and computing the `nd_range` depends on a lot of arguments (such as the total dynamic local mem usage), it was decided to use the `rangeless` approach (safer).

**NOTE**

Some SYCL features such as `sycl::stream` are unavailable in `cooperative` kernels launches because of the synchronizations it requires.

See the `device_latch` from _Data Parallel C++_, p528.

## Future extensions ?

Some of these would maybe require the usual range:

- `sycl::launch::half_occupancy` so we can expect to run two kernels in parallel?
- `sycl::launch::real_time` dependencies recursively become `real_time` too and bypass pending DAG kernels?
- `sycl::launch::lazy` waits for a `.wait()` call on the resulting event to be launched?
- `sycl::launch::oversubscribe` 10x occupancy?
- `sycl::launch::min_occupancy` (low_priority?) launches on a single compute unit to leave room for other kernels? maybe hint to OS scheduler on CPU.

### Code for the `occupancy_range_adapter`

```C++
template<typename func> static inline void occupancy_range_adapter(size_t size, sycl::nd_item<1>& item, func&& kernel) {
    const size_t launched_work_groups = item.get_group_range(0);
    const size_t reqd_items_per_group = (size + launched_work_groups - 1) / launched_work_groups;
    const size_t launched_work_items_per_group = item.get_local_range(0);
    for (size_t local_worker_offset = item.get_local_id(0); local_worker_offset < reqd_items_per_group; local_worker_offset += launched_work_items_per_group) {
        const size_t global_id = local_worker_offset + reqd_items_per_group * item.get_group(0);
        if (global_id < size) { kernel(global_id); }
    }
}
```
