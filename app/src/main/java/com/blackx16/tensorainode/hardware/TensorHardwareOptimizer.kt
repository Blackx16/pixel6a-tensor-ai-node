package com.blackx16.tensorainode.hardware

import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
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
    val isApproachingAnonSwapLimit: Boolean
        get() = (rssAnonMb + vmSwapMb) >= (anonSwapCeilingMb * 80 / 100)
}

data class NetworkEndpointInfo(
    val interfaceName: String,
    val ipAddress: String,
    val protocolLabel: String,
    val isVpnTunnel: Boolean
)

/**
 * Hardware, Memory, and Multi-VPN Network Optimizer for the Google Pixel 6a (`bluejay`).
 */
class TensorHardwareOptimizer(private val context: Context) {

    companion object {
        const val TENSOR_BIG_CORE_THREADS = 4
        const val TENSOR_IO_CORE_THREADS = 2
        const val TARGET_TOKEN_DURATION_NANOS = 25_000_000L
        const val PIXEL6A_SAFE_ANON_SWAP_CEILING_MB = 2250L

        /**
         * WHY: Standard Java `Inet4Address.isSiteLocalAddress()` ONLY checks RFC 1918 ranges
         * (10/8, 172.16/12, 192.168/16) and returns FALSE for Tailscale's RFC 6598 CGNAT range
         * (`100.64.0.0/10`, i.e., `100.64.x.x` .. `100.127.x.x`). Explicitly checking raw octets
         * allows our server to detect Tailscale and ZeroTier CGNAT VPN addresses on `tun0`.
         */
        fun isCgnatIpv4(addr: Inet4Address): Boolean {
            val raw = addr.address
            if (raw.size != 4) return false
            val first = raw[0].toInt() and 0xFF
            val second = raw[1].toInt() and 0xFF
            return first == 100 && (second in 64..127)
        }

        fun isVpnInterfaceName(name: String): Boolean {
            val n = name.lowercase()
            return n.startsWith("tun") ||
                n.startsWith("wg") ||
                n.startsWith("tailscale") ||
                n.startsWith("zt") ||
                n.startsWith("tap") ||
                n.startsWith("ppp")
        }
    }

    private val mappedHandles = ConcurrentHashMap<String, Long>()
    private var hintSession: PerformanceHintManager.Session? = null
    private val bigCoreThreadIds = ConcurrentHashMap.newKeySet<Int>()

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
     * WHY: Checks Android's `ConnectivityManager` for `NetworkCapabilities.TRANSPORT_VPN`
     * without claiming `VpnService` ourselves (since Android only allows 1 active VpnService per
     * user profile, claiming it would disconnect Tailscale/WireGuard/OpenVPN).
     */
    fun isSystemVpnTransportActive(): Boolean {
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return false
            cm.allNetworks.any { net ->
                cm.getNetworkCapabilities(net)?.hasTransport(NetworkCapabilities.TRANSPORT_VPN) == true
            }
        } catch (_: Throwable) {
            false
        }
    }

    /**
     * Enumerates all active VPN (`tun0`, `wg0`, `tailscale0`, `zt*`) and LAN (`wlan0`) IPv4 endpoints.
     */
    fun scanAllNetworkEndpoints(): List<NetworkEndpointInfo> {
        val results = mutableListOf<NetworkEndpointInfo>()
        try {
            val interfaces = NetworkInterface.getNetworkInterfaces()?.toList().orEmpty()
            for (iface in interfaces) {
                if (!iface.isUp || iface.isLoopback) continue
                val ifName = iface.name ?: continue
                // Skip cellular rmnet/ccmni WAN interfaces unless they carry a VPN/CGNAT tunnel
                if (ifName.startsWith("rmnet") || ifName.startsWith("ccmni") || ifName.startsWith("dummy")) {
                    continue
                }

                for (addr in iface.inetAddresses) {
                    if (addr is Inet4Address && !addr.isLoopbackAddress) {
                        val ip = addr.hostAddress ?: continue
                        val cgnat = isCgnatIpv4(addr)
                        val vpnIf = isVpnInterfaceName(ifName) || cgnat
                        if (!addr.isSiteLocalAddress && !cgnat && !vpnIf) continue

                        val label = when {
                            cgnat -> "Tailscale / CGNAT Mesh ($ifName)"
                            ifName.startsWith("wg") -> "WireGuard Tunnel ($ifName)"
                            ifName.startsWith("zt") -> "ZeroTier Mesh ($ifName)"
                            ifName.startsWith("tun") || ifName.startsWith("tap") -> "VPN Tunnel ($ifName • WireGuard/OpenVPN/Tailscale)"
                            ifName.startsWith("wlan") -> "Local Wi-Fi ($ifName)"
                            else -> "LAN Interface ($ifName)"
                        }
                        results.add(
                            NetworkEndpointInfo(
                                interfaceName = ifName,
                                ipAddress = ip,
                                protocolLabel = label,
                                isVpnTunnel = vpnIf
                            )
                        )
                    }
                }
            }
        } catch (_: Throwable) {
        }
        // Sort so VPN tunnels (Tailscale / WireGuard / OpenVPN) appear first, followed by wlan0
        return results.sortedWith(
            compareByDescending<NetworkEndpointInfo> { it.isVpnTunnel }
                .thenBy { !it.interfaceName.startsWith("wlan") }
        )
    }

    /**
     * Returns the best host IP (preferring an active Tailscale/WireGuard/OpenVPN tunnel IP if connected,
     * otherwise the local Wi-Fi IP).
     */
    fun getLocalWifiIpAddress(): String {
        val endpoints = scanAllNetworkEndpoints()
        return endpoints.firstOrNull()?.ipAddress ?: "127.0.0.1"
    }

    fun getVpnAndWifiSummary(port: Int = 8080): String {
        val endpoints = scanAllNetworkEndpoints()
        if (endpoints.isEmpty()) {
            return "No VPN or Wi-Fi detected (127.0.0.1:$port)"
        }
        return endpoints.joinToString("\n") { ep ->
            val tag = if (ep.isVpnTunnel) "[VPN TUNNEL]" else "[LOCAL WI-FI]"
            "$tag ${ep.protocolLabel}: http://${ep.ipAddress}:$port/v1"
        }
    }
}
