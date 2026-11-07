#include <jni.h>
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdint>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "TensorGS101", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "TensorGS101", __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n")
#define LOGW(...) fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#endif

/**
 * WHY: The Google Pixel 6a (Google Tensor GS101) has an asymmetric 2+2+4 CPU topology:
 *   - cpu0..cpu3 (0x0F): 4x ARM Cortex-A55 @ 1.80 GHz (in-order efficiency cores)
 *   - cpu4..cpu5 (0x30): 2x ARM Cortex-A76 @ 2.25 GHz (out-of-order performance cores)
 *   - cpu6..cpu7 (0xC0): 2x ARM Cortex-X1  @ 2.80 GHz (out-of-order prime cores)
 *
 * When running barrier-synchronized matrix multiplication (GEMV/GEMM), scheduling threads onto
 * the in-order Cortex-A55 cores causes the fast Cortex-X1 and Cortex-A76 cores to stall at
 * synchronization barriers. Pinning inference workers strictly to mask 0xF0 (cpu4..cpu7) and
 * HTTP/disk I/O threads to mask 0x0F (cpu0..cpu3) eliminates barrier stalls and matches the
 * 4-thread LPDDR5/LPDDR4X memory bandwidth sweet spot.
 */
static constexpr uint32_t TENSOR_GS101_BIG_CORES_MASK    = 0xF0; // cpu4, cpu5, cpu6, cpu7
static constexpr uint32_t TENSOR_GS101_PRIME_CORES_MASK  = 0xC0; // cpu6, cpu7
static constexpr uint32_t TENSOR_GS101_LITTLE_CORES_MASK = 0x0F; // cpu0, cpu1, cpu2, cpu3

struct MappedModelRegion {
    void* addr;
    size_t length;
    int fd;
};

static bool applyCpuMask(uint32_t mask) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu = 0; cpu < 8; ++cpu) {
        if (mask & (1u << cpu)) {
            CPU_SET(cpu, &cpuset);
        }
    }
    pid_t tid = gettid();
    int rc = sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        LOGW("sched_setaffinity(tid=%d, mask=0x%02x) returned %d", tid, mask, rc);
        return false;
    }
    LOGI("Pinned thread tid=%d to Tensor GS101 CPU mask 0x%02x", tid, mask);
    return true;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_pinCurrentThreadToBigCores(
        JNIEnv* /*env*/, jobject /*thiz*/) {
    return applyCpuMask(TENSOR_GS101_BIG_CORES_MASK) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_pinCurrentThreadToPrimeCores(
        JNIEnv* /*env*/, jobject /*thiz*/) {
    return applyCpuMask(TENSOR_GS101_PRIME_CORES_MASK) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_pinCurrentThreadToLittleCores(
        JNIEnv* /*env*/, jobject /*thiz*/) {
    return applyCpuMask(TENSOR_GS101_LITTLE_CORES_MASK) ? JNI_TRUE : JNI_FALSE;
}

/**
 * WHY: Query the kernel page size dynamically via sysconf(_SC_PAGESIZE) instead of hardcoding 4096,
 * ensuring compatibility with both 4 KB kernels (6.1.172-android14-11) and 16 KB page modes on Android 17.
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_getSystemPageSize(
        JNIEnv* /*env*/, jobject /*thiz*/) {
    return static_cast<jint>(sysconf(_SC_PAGESIZE));
}

/**
 * WHY: Android 17 introduces a strict per-app anonymous memory + ZRAM swap ceiling (MemoryLimiter:AnonSwap).
 * On a 6 GB RAM Pixel 6a, loading a 1-2 GB model into anonymous heap (malloc/RssAnon) risks immediate
 * termination. Mapping weights read-only with mmap(PROT_READ, MAP_SHARED) places them in the file-backed
 * Linux page cache (RssFile), which the kernel can cleanly reclaim and re-fault from UFS 3.1 storage.
 */
extern "C" JNIEXPORT jlong JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_mmapModelReadOnly(
        JNIEnv* env, jobject /*thiz*/, jstring filePath) {
    const char* path = env->GetStringUTFChars(filePath, nullptr);
    if (!path) return 0L;

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    env->ReleaseStringUTFChars(filePath, path);
    if (fd < 0) {
        return 0L;
    }

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return 0L;
    }

    size_t length = static_cast<size_t>(st.st_size);
    void* addr = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        return 0L;
    }

    // Hint the Linux kernel on Pixel 6a (UFS 3.1) for sequential read-ahead during weight loading
    madvise(addr, length, MADV_SEQUENTIAL);

    auto* region = new MappedModelRegion{addr, length, fd};
    return reinterpret_cast<jlong>(region);
}

/**
 * WHY: When Android 17 dispatches onTrimMemory(TRIM_MEMORY_UI_HIDDEN / TRIM_MEMORY_BACKGROUND),
 * madvise(MADV_DONTNEED) immediately frees physical RAM pages for the OS while keeping the virtual
 * address mapping valid for instant fault-in on the next request.
 */
extern "C" JNIEXPORT jboolean JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_evictMmapPages(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    if (handle == 0L) return JNI_FALSE;
    auto* region = reinterpret_cast<MappedModelRegion*>(handle);
    int rc = madvise(region->addr, region->length, MADV_DONTNEED);
    return (rc == 0) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_unmapModel(
        JNIEnv* /*env*/, jobject /*thiz*/, jlong handle) {
    if (handle == 0L) return;
    auto* region = reinterpret_cast<MappedModelRegion*>(handle);
    if (region->addr && region->addr != MAP_FAILED) {
        munmap(region->addr, region->length);
    }
    if (region->fd >= 0) {
        close(region->fd);
    }
    delete region;
}

/**
 * WHY: Parse /proc/self/status directly to separate RssAnon (counted against Android 17's AnonSwap killer)
 * from RssFile (reclaimable mmap model weights).
 * Returns long[4]: [VmRSS_KB, RssAnon_KB, RssFile_KB, VmSwap_KB].
 */
extern "C" JNIEXPORT jlongArray JNICALL
Java_com_blackx16_tensorainode_hardware_TensorNativeBridge_readProcessMemoryKb(
        JNIEnv* env, jobject /*thiz*/) {
    jlong stats[4] = {0, 0, 0, 0};
    std::ifstream statusFile("/proc/self/status");
    std::string line;
    while (std::getline(statusFile, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            iss >> stats[0];
        } else if (line.rfind("RssAnon:", 0) == 0) {
            std::istringstream iss(line.substr(8));
            iss >> stats[1];
        } else if (line.rfind("RssFile:", 0) == 0) {
            std::istringstream iss(line.substr(8));
            iss >> stats[2];
        } else if (line.rfind("VmSwap:", 0) == 0) {
            std::istringstream iss(line.substr(7));
            iss >> stats[3];
        }
    }
    jlongArray result = env->NewLongArray(4);
    if (result) {
        env->SetLongArrayRegion(result, 0, 4, stats);
    }
    return result;
}
