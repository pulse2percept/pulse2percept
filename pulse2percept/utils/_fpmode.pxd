# Subnormal-flushing control for the temporal models' inner loops.
#
# Every temporal model in `pulse2percept.models` is a cascade of leaky
# integrators stepped at `dt`, and between pulses each one decays exponentially
# toward zero. A pulse train leaves them decaying for a long time: at the
# default dt=5e-3 ms, a 6 Hz train puts ~33,000 steps between one pulse and the
# next, which for a tau of half a millisecond is several hundred time
# constants. Partway down that decay the state enters the subnormal range
# (|x| < 1.2e-38 in float32), where x86 hands the operation to microcode and
# takes roughly two orders of magnitude longer than the same arithmetic on
# normal floats. It does not stay there long -- a few percent of steps -- but
# at 100x each that is enough to dominate the run: measured on the Horsager
# 2009 kernel with an Argus II pulse train, the loop runs ~9x slower than the
# identical loop on values that never go subnormal.
#
# Flushing them to zero costs nothing that matters. These are magnitudes below
# 1e-38 feeding a percept of order 1e-2, so the alternative to zero is not a
# more accurate answer, it is the same answer reached slowly.
#
# The mode lives in a per-thread control register, so it has to be set inside
# the parallel region rather than around it, and restored afterwards so that
# importing this library does not silently change the floating-point behavior
# of everything else in the process.

cdef extern from *:
    """
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
        defined(_M_IX86)
      #include <xmmintrin.h>
      /* MXCSR bit 15 (FTZ) flushes subnormal results to zero; bit 6 (DAZ)
         treats subnormal operands as zero. Both are present on every x86-64
         CPU, and on 32-bit x86 back to Prescott. */
      static CYTHON_INLINE unsigned long long p2p_denormals_off(void) {
          unsigned int prev = _mm_getcsr();
          _mm_setcsr(prev | 0x8000u | 0x0040u);
          return (unsigned long long) prev;
      }
      static CYTHON_INLINE void p2p_fpmode_restore(unsigned long long prev) {
          _mm_setcsr((unsigned int) prev);
      }
    #elif defined(__aarch64__) && defined(__GNUC__)
      /* FPCR bit 24 (FZ) is the AArch64 equivalent. Save and restore the whole
         register rather than just the bit, so nothing else in it is disturbed. */
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
      /* Unknown architecture: leave the floating-point mode alone. The models
         are correct either way; only the speed differs. */
      static CYTHON_INLINE unsigned long long p2p_denormals_off(void) {
          return 0ULL;
      }
      static CYTHON_INLINE void p2p_fpmode_restore(unsigned long long prev) {
          (void) prev;
      }
    #endif
    """
    # Stop subnormal results from being computed as subnormals, returning the
    # previous mode of the calling thread for `c_fpmode_restore`.
    unsigned long long c_denormals_off "p2p_denormals_off" () noexcept nogil
    void c_fpmode_restore "p2p_fpmode_restore" (
        unsigned long long prev) noexcept nogil
