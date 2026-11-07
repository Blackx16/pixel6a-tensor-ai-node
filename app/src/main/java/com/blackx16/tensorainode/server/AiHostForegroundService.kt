package com.blackx16.tensorainode.server

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.net.wifi.WifiManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import com.blackx16.tensorainode.engine.UnifiedAiRouter
import com.blackx16.tensorainode.hardware.TensorHardwareOptimizer
import com.blackx16.tensorainode.models.ModelDownloadManager
import com.blackx16.tensorainode.ui.MainActivity
import java.io.File

/**
 * Android 17 (API 37) compliant Foreground Service (`connectedDevice|specialUse`).
 *
 * WHY: Keeps the local Wi-Fi OpenAI server and Tensor GS101 inference engine active when the screen
 * turns off, holding a `WIFI_MODE_FULL_LOW_LATENCY` lock so token streaming latency over LAN stays low.
 */
class AiHostForegroundService : Service() {

    companion object {
        const val CHANNEL_ID = "tensor_ai_host_channel"
        const val NOTIFICATION_ID = 1701
        const val ACTION_START_SERVER = "com.blackx16.tensorainode.START_SERVER"
        const val ACTION_STOP_SERVER = "com.blackx16.tensorainode.STOP_SERVER"
        const val EXTRA_API_KEY = "extra_api_key"
        const val EXTRA_PORT = "extra_port"
    }

    inner class LocalBinder : Binder() {
        fun getService(): AiHostForegroundService = this@AiHostForegroundService
    }

    private val binder = LocalBinder()
    lateinit var optimizer: TensorHardwareOptimizer
        private set
    lateinit var downloadManager: ModelDownloadManager
        private set
    lateinit var router: UnifiedAiRouter
        private set
    var httpServer: OpenAiHttpServer? = null
        private set

    private var wifiLock: WifiManager.WifiLock? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        optimizer = TensorHardwareOptimizer(applicationContext)
        downloadManager = ModelDownloadManager(
            modelsRootDir = File(filesDir, "models"),
            ioDispatcher = optimizer.networkIoDispatcher
        )
        router = UnifiedAiRouter(applicationContext, optimizer, downloadManager)
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP_SERVER -> {
                stopOpenAiServer()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                val key = intent?.getStringExtra(EXTRA_API_KEY)?.ifBlank { "sk-pixel6a-tensor" }
                    ?: "sk-pixel6a-tensor"
                val port = intent?.getIntExtra(EXTRA_PORT, 8080) ?: 8080
                startOpenAiServer(key, port)
            }
        }
        return START_STICKY
    }

    fun startOpenAiServer(apiKey: String, port: Int = 8080) {
        acquireLocks()
        val existing = httpServer
        if (existing != null && existing.isRunning) {
            existing.apiKey = apiKey
        } else {
            val srv = OpenAiHttpServer(
                router = router,
                apiKey = apiKey,
                port = port,
                ioDispatcher = optimizer.networkIoDispatcher
            )
            srv.start()
            httpServer = srv
        }

        val wifiIp = optimizer.getLocalWifiIpAddress()
        val notification = buildNotification("Hosting at http://$wifiIp:$port/v1 (Bearer key active)")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE or
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    fun stopOpenAiServer() {
        httpServer?.stop()
        httpServer = null
        releaseLocks()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        optimizer.handleTrimMemory(level) {
            router.evictAllIdleCaches()
        }
    }

    override fun onDestroy() {
        stopOpenAiServer()
        router.evictAllIdleCaches()
        optimizer.releaseAllMappings()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder = binder

    private fun acquireLocks() {
        try {
            val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager
            if (wifiLock == null) {
                wifiLock = wm?.createWifiLock(
                    WifiManager.WIFI_MODE_FULL_LOW_LATENCY,
                    "TensorAINode:WifiLock"
                )?.apply {
                    setReferenceCounted(false)
                    acquire()
                }
            }
            val pm = applicationContext.getSystemService(Context.POWER_SERVICE) as? PowerManager
            if (wakeLock == null) {
                wakeLock = pm?.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    "TensorAINode:CpuWakeLock"
                )?.apply {
                    setReferenceCounted(false)
                    acquire(6 * 60 * 60 * 1000L)
                }
            }
        } catch (_: Throwable) {
        }
    }

    private fun releaseLocks() {
        try {
            wifiLock?.takeIf { it.isHeld }?.release()
            wakeLock?.takeIf { it.isHeld }?.release()
        } catch (_: Throwable) {
        }
        wifiLock = null
        wakeLock = null
    }

    private fun createNotificationChannel() {
        val nm = getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Pixel 6a Local AI Wi-Fi Host",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Keeps the local OpenAI Wi-Fi server active on your LAN."
        }
        nm.createNotificationChannel(channel)
    }

    private fun buildNotification(contentText: String): Notification {
        val launchIntent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            launchIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("TensorAI Node (Pixel 6a)")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.stat_sys_upload_done)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }
}
