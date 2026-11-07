package com.blackx16.tensorainode.hardware

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import java.io.File
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

data class MemoryTelemetry(
    val totalDeviceRamMb: Long,
    val availableDeviceRamMb: Long,
    val vmRssMb: Long,
    val rssAnonMb: Long,
    val rssFileMmapMb: Long,
    val vmSwapMb: Long,
    val pageSizeBytes: Int,
    val anonSwapCeilingMb: Long
) {
    // WHY: Trigger proactive secondary model eviction if RssAnon + Swap reaches 80% of the safe ceiling.
    val isApproachingAnonSwapLimit: Boolean
        get() = (rssAnonMb + vmSwapMb) >= (anonSwapCeilingMb * 80 / 100)
}

/**
 * Hardware and OS optimizer specifically engineered for the Google Pixel 6a (`bluejay`):
 * - SoC: Google Tensor GS101 (2x 2.80 GHz Cortex-X1 + 2x 2.25 GHz Cortex-A76 + 4x 1.80 GHz Cortex-A55)
 * - GPU: Mali-G78 MP20 (20 cores)
 * - RAM: 6 GB LPDDR5/LPDDR4X (~5.43 GB usable)
 * - OS: Android 17 (API 37, Build DP11.261016.003, Kernel 6.1.172-android14-11)
 */
class TensorHardwareOptimizer(private val context: Context) {

    companion object {
        // WHY: Use exactly 4 threads for inference to match the 2x Cortex-X1 + 2x Cortex-A76 big cores.
        // Adding more threads spills onto the in-order Cortex-A55 cores and causes barrier stalls.
        const val TENSOR_BIG_CORE_THREADS = 4
        const val TENSOR_IO_CORE_THREADS = 2

        // WHY: 25ms target per token step (40 tok/s) instructs Android's ADPF scheduler to ramp up
        // Cortex-X1/A76 DVFS clocks immediately without the default ~200ms governor latency.
        const val TARGET_TOKEN_DURATION_NANOS = 25_000_000L

        // WHY: On a 6 GB Pixel 6a where Android 17 uses ~2.5-3.0 GB, capping anonymous heap + swap
        // at 2250 MB prevents Android 17's MemoryLimiter:AnonSwap killer from terminating the app.
        const val PIXEL6A_SAFE_ANON_SWAP_CEILING_MB = 2250L
    }

    private val mappedHandles = ConcurrentHashMap<String, Long>()
    private var hintSession: PerformanceHintManager.Session? = null
    private val bigCoreThreadIds = ConcurrentHashMap.newKeySet<Int>()

    /**
     * WHY: Dedicated 4-thread pool pinned via sched_setaffinity to CPU mask 0xF0 (cpu4..cpu7)
     * and registered with Android's PerformanceHintManager.
     */
    val inferenceDispatcher: CoroutineDispatcher = run {
        val index = AtomicInteger(0)
        Executors.newFixedThreadPool(TENSOR_BIG_CORE_THREADS) { runnable ->
            Thread({
                Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_DISPLAY)
                TensorNativeBridge.safePinToBigCores()
                bigCoreThreadIds.add(Process.myTid())
                ensurePerformanceHintSession()
                runnable.run()
            }, "TensorGS101-BigCore-${index.getAndIncrement()}").apply {
                isDaemon = true
            }
        }.asCoroutineDispatcher()
    }

    /**
     * WHY: Dedicated 2-thread I/O pool pinned to CPU mask 0x0F (Cortex-A55 efficiency cores cpu0..cpu3)
     * so HTTP socket traffic and file downloads never contend with LLM token generation on big cores.
     */
    val networkIoDispatcher: CoroutineDispatcher = run {
        val index = AtomicInteger(0)
        Executors.newFixedThreadPool(TENSOR_IO_CORE_THREADS) { runnable ->
            Thread({
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
                TensorNativeBridge.safePinToLittleCores()
                runnable.run()
            }, "TensorGS101-LittleCore-IO-${index.getAndIncrement()}").apply {
                isDaemon = true
            }
        }.asCoroutineDispatcher()
    }

    @Synchronized
    private fun ensurePerformanceHintSession() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && hintSession == null && bigCoreThreadIds.isNotEmpty()) {
            try {
                val phm = context.getSystemService(PerformanceHintManager::class.java)
                val tids = bigCoreThreadIds.toIntArray()
                hintSession = phm?.createHintSession(tids, TARGET_TOKEN_DURATION_NANOS)
            } catch (_: Throwable) {
            }
        }
    }

    fun reportTokenStepDuration(actualDurationNanos: Long) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                hintSession?.reportActualWorkDuration(actualDurationNanos.coerceAtLeast(1_000_000L))
            } catch (_: Throwable) {
            }
        }
    }

    fun mmapModelWeights(modelFile: File): Boolean {
        if (!modelFile.exists() || !TensorNativeBridge.isNativeLoaded) return false
        val path = modelFile.absolutePath
        if (mappedHandles.containsKey(path)) return true
        val handle = TensorNativeBridge.mmapModelReadOnly(path)
        if (handle != 0L) {
            mappedHandles[path] = handle
            return true
        }
        return false
    }

    /**
     * WHY: Android 14-17 deprecated TRIM_MEMORY_RUNNING_* callbacks and only dispatches
     * TRIM_MEMORY_UI_HIDDEN and TRIM_MEMORY_BACKGROUND. We respond to those exact signals
     * by advising the kernel to drop clean mmap weight pages and evicting idle sessions.
     */
    fun handleTrimMemory(level: Int, onEvictIdleEngines: () -> Unit) {
        if (level >= ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) {
            mappedHandles.values.forEach { handle ->
                TensorNativeBridge.evictMmapPages(handle)
            }
        }
        if (level >= ComponentCallbacks2.TRIM_MEMORY_BACKGROUND) {
            onEvictIdleEngines()
        }
    }

    fun releaseAllMappings() {
        mappedHandles.values.forEach { handle ->
            TensorNativeBridge.unmapModel(handle)
        }
        mappedHandles.clear()
        try {
            hintSession?.close()
        } catch (_: Throwable) {
        }
        hintSession = null
    }

    fun readMemoryTelemetry(): MemoryTelemetry {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        val memInfo = ActivityManager.MemoryInfo()
        am?.getMemoryInfo(memInfo)

        val procStats = if (TensorNativeBridge.isNativeLoaded) {
            TensorNativeBridge.readProcessMemoryKb()
        } else {
            longArrayOf(0L, 0L, 0L, 0L)
        }
        val pageSize = if (TensorNativeBridge.isNativeLoaded) {
            TensorNativeBridge.getSystemPageSize()
        } else {
            4096
        }

        return MemoryTelemetry(
            totalDeviceRamMb = if (memInfo.totalMem > 0) memInfo.totalMem / (1024 * 1024) else 5850L,
            availableDeviceRamMb = if (memInfo.availMem > 0) memInfo.availMem / (1024 * 1024) else 3200L,
            vmRssMb = procStats[0] / 1024,
            rssAnonMb = procStats[1] / 1024,
            rssFileMmapMb = procStats[2] / 1024,
            vmSwapMb = procStats[3] / 1024,
            pageSizeBytes = pageSize,
            anonSwapCeilingMb = PIXEL6A_SAFE_ANON_SWAP_CEILING_MB
        )
    }

    /**
     * WHY: Inspects network interfaces prioritizing `wlan0` so the user and LAN devices
     * immediately get the exact local Wi-Fi IPv4 address (e.g. 192.168.1.x) for OpenAI clients.
     */
    fun getLocalWifiIpAddress(): String {
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()?.toList().orEmpty()
            val prioritized = interfaces.sortedBy { iface ->
                when {
                    iface.name.startsWith("wlan") -> 0
                    iface.name.startsWith("ap") -> 1
                    iface.name.startsWith("eth") -> 2
                    else -> 3
                }
            }
            for (iface in prioritized) {
                if (!iface.isUp || iface.isLoopback) continue
                for (addr in iface.inetAddresses) {
                    if (addr is Inet4Address && !addr.isLoopbackAddress && addr.isSiteLocalAddress) {
                        return addr.hostAddress ?: continue
                    }
                }
            }
        } catch (_: Throwable) {
        }
        return "127.0.0.1"
    }
}
