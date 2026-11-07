package com.blackx16.tensorainode.hardware

/**
 * JNI bridge for Google Tensor GS101 low-level CPU affinity, page size inspection,
 * and file-backed mmap weight management on Android 17 (API 37).
 */
object TensorNativeBridge {
    // WHY: Android 17 (API 37) forbids mutating static final fields via reflection or JNI;
    // keeping mutable runtime flags as non-final @Volatile properties ensures strict API 37 compliance.
    @Volatile
    var isNativeLoaded: Boolean = false
        private set

    init {
        try {
            System.loadLibrary("tensor_affinity")
            isNativeLoaded = true
        } catch (_: UnsatisfiedLinkError) {
            isNativeLoaded = false
        }
    }

    external fun pinCurrentThreadToBigCores(): Boolean
    external fun pinCurrentThreadToPrimeCores(): Boolean
    external fun pinCurrentThreadToLittleCores(): Boolean
    external fun getSystemPageSize(): Int
    external fun mmapModelReadOnly(filePath: String): Long
    external fun evictMmapPages(handle: Long): Boolean
    external fun unmapModel(handle: Long)
    external fun readProcessMemoryKb(): LongArray

    fun safePinToBigCores(): Boolean =
        if (isNativeLoaded) pinCurrentThreadToBigCores() else false

    fun safePinToLittleCores(): Boolean =
        if (isNativeLoaded) pinCurrentThreadToLittleCores() else false
}
