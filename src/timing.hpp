#ifndef TIMING_HPP
#define TIMING_HPP

// Check if OpenMP is enabled by the compiler
#if defined(_OPENMP)
    // If OpenMP is available, include its header
    #include <omp.h>
#else
    // If OpenMP is not available, include the C++ chrono library for an alternative
    #include <chrono>
#endif

/**
 * @brief A wrapper function to get the current wall-clock time in seconds.
 * 
 * This function uses omp_get_wtime() if compiled with OpenMP support.
 * Otherwise, it falls back to using std::chrono::steady_clock for high-resolution timing
 * in a serial context.
 * 
 * @return double The current time in seconds since an arbitrary epoch.
 */
inline double get_wtime() {
#if defined(_OPENMP)
    // Use the OpenMP high-resolution timer
    return omp_get_wtime();
#else
    // Fallback to C++11 chrono library for the serial version.
    // steady_clock is used as it is not affected by system time changes.
    auto now = std::chrono::steady_clock::now();
    // Convert the time point to a duration in seconds (as a double).
    return std::chrono::duration<double>(now.time_since_epoch()).count();
#endif
}

inline int get_thread_num() {
#if defined(_OPENMP)
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline int get_num_threads() {
#if defined(_OPENMP)
    return omp_get_num_threads();
#else
    return 1;
#endif
}


#endif // TIMING_HPP