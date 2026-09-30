# Subnormal flushing for the temporal models' inner loops.
#
# Temporal models are leaky-integrator cascades stepped at `dt`, decaying
# toward zero between pulses (at dt=5e-3 ms, a 6 Hz train has ~33,000 steps
# between pulses). Part of that decay is in the float32 subnormal range
# (|x| < 1.2e-38), where x86 arithmetic is ~100x slower. On the Horsager 2009
# kernel with an Argus II pulse train, this made the loop ~9x slower.
# Flushing these values to zero does not change percepts (order 1e-2).
#
# The mode is a per-thread register: set it inside the parallel region, and
# restore it afterwards so other code in the process is unaffected.

cdef extern from *:
    """
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
        defined(_M_IX86)
      #include <xmmintrin.h>
      /* MXCSR bit 15 (FTZ) flushes subnormal results to zero; bit 6 (DAZ)
         treats subnormal operands as zero. Available on all x86-64 CPUs and
         32-bit x86 since Prescott. */
      static CYTHON_INLINE unsigned long long p2p_denormals_off(void) {
          unsigned int prev = _mm_getcsr();
          _mm_setcsr(prev | 0x8000u | 0x0040u);
          return (unsigned long long) prev;
      }
      static CYTHON_INLINE void p2p_fpmode_restore(unsigned long long prev) {
          _mm_setcsr((unsigned int) prev);
      }
    #elif defined(__aarch64__) && defined(__GNUC__)
      /* FPCR bit 24 (FZ) is the AArch64 equivalent. The whole register is
         saved and restored. */
      static CYTHON_INLINE unsigned long long p2p_denormals_off(void) {
          unsigned long long prev;
          __asm__ __volatile__("mrs %0, fpcr" : "=r" (prev));
          __asm__ __volatile__("msr fpcr, %0" : : "r" (prev | (1ULL << 24)));
          return prev;
      }
      static CYTHON_INLINE void p2p_fpmode_restore(unsigned long long prev) {
          __asm__ __volatile__("msr fpcr, %0" : : "r" (prev));
      }
    #else
      /* Unknown architecture: floating-point mode is unchanged (slower, same
         results). */
      static CYTHON_INLINE unsigned long long p2p_denormals_off(void) {
          return 0ULL;
      }
      static CYTHON_INLINE void p2p_fpmode_restore(unsigned long long prev) {
          (void) prev;
      }
    #endif
    """
    # Flush subnormals to zero; returns the thread's previous mode for
    # `c_fpmode_restore`:
    unsigned long long c_denormals_off "p2p_denormals_off" () noexcept nogil
    void c_fpmode_restore "p2p_fpmode_restore" (
        unsigned long long prev) noexcept nogil
