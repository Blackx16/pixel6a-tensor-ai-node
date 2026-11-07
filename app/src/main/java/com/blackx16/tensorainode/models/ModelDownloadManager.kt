package com.blackx16.tensorainode.models

import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL

data class ModelDownloadProgress(
    val modelId: String,
    val bytesDownloaded: Long,
    val totalBytes: Long,
    val isDownloading: Boolean,
    val isInstalled: Boolean,
    val error: String? = null
) {
    val percent: Int
        get() = if (totalBytes > 0) ((bytesDownloaded * 100L) / totalBytes).toInt().coerceIn(0, 100) else 0
}

/**
 * Post-install web model downloader with HTTP Range resume support, zero-heap streaming,
 * optional Hugging Face Bearer token authentication, and custom URL installation.
 */
class ModelDownloadManager(
    private val modelsRootDir: File,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) {
    private val _progressMap = MutableStateFlow<Map<String, ModelDownloadProgress>>(emptyMap())
    val progressMap: StateFlow<Map<String, ModelDownloadProgress>> = _progressMap.asStateFlow()

    private val customModels = mutableListOf<AiModelSpec>()

    init {
        if (!modelsRootDir.exists()) {
            modelsRootDir.mkdirs()
        }
        refreshInstallationStates()
    }

    fun getAllModels(): List<AiModelSpec> = ModelCatalog.BUILTIN_MODELS + customModels

    fun getModelDir(spec: AiModelSpec): File {
        val firstRel = spec.files.first().relativePath
        val parentFolder = firstRel.substringBefore("/", missingDelimiterValue = spec.id)
        return File(modelsRootDir, parentFolder)
    }

    fun getPrimaryModelFile(spec: AiModelSpec): File =
        File(modelsRootDir, spec.files.first().relativePath)

    fun isModelInstalled(spec: AiModelSpec): Boolean =
        spec.files.all { fileSpec ->
            val f = File(modelsRootDir, fileSpec.relativePath)
            f.exists() && f.length() > 0L
        }

    fun refreshInstallationStates() {
        val updated = mutableMapOf<String, ModelDownloadProgress>()
        for (spec in getAllModels()) {
            val installed = isModelInstalled(spec)
            val existingBytes = spec.files.sumOf {
                val f = File(modelsRootDir, it.relativePath)
                if (f.exists()) f.length() else 0L
            }
            updated[spec.id] = ModelDownloadProgress(
                modelId = spec.id,
                bytesDownloaded = existingBytes,
                totalBytes = spec.totalDownloadBytes,
                isDownloading = false,
                isInstalled = installed
            )
        }
        _progressMap.value = updated
    }

    /**
     * WHY: Allows the user to plug in any additional `.litertlm`, `.task`, `.tflite`, or `.onnx`
     * model URL from the web after installation without rebuilding the app.
     */
    fun registerCustomModelUrl(url: String, customId: String? = null): AiModelSpec {
        val cleanFileName = url.substringAfterLast("/").substringBefore("?").ifBlank { "custom_model.litertlm" }
        val runtime = if (cleanFileName.endsWith(".onnx", ignoreCase = true)) {
            RuntimeBackend.ONNX
        } else {
            RuntimeBackend.LITERT
        }
        val id = customId?.takeIf { it.isNotBlank() } ?: cleanFileName.substringBeforeLast(".")
        val spec = AiModelSpec(
            id = id,
            title = "Custom: $cleanFileName",
            runtime = runtime,
            capabilities = setOf(ModelCapability.TEXT, ModelCapability.IMAGE_VISION, ModelCapability.EMBEDDINGS),
            quantization = "Custom Web Download",
            estimatedRamMb = 1024,
            description = "Custom web-downloaded model from $url",
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "custom-$id/$cleanFileName",
                    downloadUrl = url,
                    expectedSizeBytes = 100_000_000L
                )
            )
        )
        customModels.removeAll { it.id == spec.id }
        customModels.add(spec)
        refreshInstallationStates()
        return spec
    }

    /**
     * WHY: Hugging Face `/resolve/main/` URLs issue cross-host 302 redirects to CDN storage, and Java's
     * default `HttpURLConnection` strips `Authorization` headers or fails on cross-protocol redirects.
     * Handling redirects manually and streaming via a 64 KB buffer to a `.part` `RandomAccessFile`
     * supports pause/resume (`Range: bytes=...`) and uses <100 KB of JVM heap even for 2 GB models.
     */
    suspend fun downloadModel(
        spec: AiModelSpec,
        hfBearerToken: String? = null,
        onProgress: ((ModelDownloadProgress) -> Unit)? = null
    ): Result<File> = withContext(ioDispatcher) {
        try {
            var cumulativeDownloaded = 0L
            val totalExpected = spec.totalDownloadBytes.coerceAtLeast(1L)

            updateState(
                ModelDownloadProgress(
                    modelId = spec.id,
                    bytesDownloaded = 0L,
                    totalBytes = totalExpected,
                    isDownloading = true,
                    isInstalled = false
                ),
                onProgress
            )

            for (fileSpec in spec.files) {
                val targetFile = File(modelsRootDir, fileSpec.relativePath)
                targetFile.parentFile?.mkdirs()
                val partFile = File(targetFile.parentFile, "${targetFile.name}.part")

                if (targetFile.exists() && targetFile.length() > 0L) {
                    cumulativeDownloaded += targetFile.length()
                    continue
                }

                val existingPartBytes = if (partFile.exists()) partFile.length() else 0L
                var currentUrl = fileSpec.downloadUrl
                var redirects = 0
                var connection: HttpURLConnection? = null

                while (redirects < 6) {
                    val conn = (URL(currentUrl).openConnection() as HttpURLConnection).apply {
                        connectTimeout = 20_000
                        readTimeout = 60_000
                        instanceFollowRedirects = false
                        setRequestProperty("User-Agent", "Pixel6a-TensorAINode/1.0 (Android 17; API 37)")
                        if (!hfBearerToken.isNullOrBlank() && currentUrl.contains("huggingface.co")) {
                            setRequestProperty("Authorization", "Bearer ${hfBearerToken.trim()}")
                        }
                        if (existingPartBytes > 0L) {
                            setRequestProperty("Range", "bytes=$existingPartBytes-")
                        }
                    }
                    val status = conn.responseCode
                    if (status in listOf(301, 302, 303, 307, 308)) {
                        val location = conn.getHeaderField("Location")
                            ?: throw IllegalStateException("Redirect missing Location header for $currentUrl")
                        currentUrl = URL(URL(currentUrl), location).toString()
                        conn.disconnect()
                        redirects++
                    } else {
                        connection = conn
                        break
                    }
                }

                val activeConn = connection ?: throw IllegalStateException("Too many redirects for ${fileSpec.downloadUrl}")
                val status = activeConn.responseCode
                if (status != HttpURLConnection.HTTP_OK && status != HttpURLConnection.HTTP_PARTIAL) {
                    activeConn.disconnect()
                    throw IllegalStateException(
                        "HTTP $status downloading ${fileSpec.relativePath}. " +
                            if (status == 401 || status == 403) "Provide a Hugging Face token if this model is gated." else ""
                    )
                }

                val appendMode = (status == HttpURLConnection.HTTP_PARTIAL && existingPartBytes > 0L)
                var fileOffset = if (appendMode) existingPartBytes else 0L

                RandomAccessFile(partFile, "rw").use { raf ->
                    if (appendMode) {
                        raf.seek(existingPartBytes)
                    } else {
                        raf.setLength(0L)
                    }
                    BufferedInputStream(activeConn.inputStream, 64 * 1024).use { input ->
                        val buffer = ByteArray(64 * 1024)
                        var read: Int
                        while (input.read(buffer).also { read = it } != -1) {
                            raf.write(buffer, 0, read)
                            fileOffset += read
                            val snap = ModelDownloadProgress(
                                modelId = spec.id,
                                bytesDownloaded = cumulativeDownloaded + fileOffset,
                                totalBytes = totalExpected,
                                isDownloading = true,
                                isInstalled = false
                            )
                            updateState(snap, onProgress)
                        }
                    }
                }
                activeConn.disconnect()

                if (targetFile.exists()) targetFile.delete()
                if (!partFile.renameTo(targetFile)) {
                    partFile.copyTo(targetFile, overwrite = true)
                    partFile.delete()
                }
                cumulativeDownloaded += targetFile.length()
            }

            val finalProgress = ModelDownloadProgress(
                modelId = spec.id,
                bytesDownloaded = cumulativeDownloaded,
                totalBytes = cumulativeDownloaded.coerceAtLeast(totalExpected),
                isDownloading = false,
                isInstalled = true
            )
            updateState(finalProgress, onProgress)
            Result.success(getPrimaryModelFile(spec))
        } catch (e: Throwable) {
            val failedProgress = ModelDownloadProgress(
                modelId = spec.id,
                bytesDownloaded = _progressMap.value[spec.id]?.bytesDownloaded ?: 0L,
                totalBytes = spec.totalDownloadBytes,
                isDownloading = false,
                isInstalled = isModelInstalled(spec),
                error = e.message ?: "Download failed"
            )
            updateState(failedProgress, onProgress)
            Result.failure(e)
        }
    }

    fun deleteModel(spec: AiModelSpec): Boolean {
        var allDeleted = true
        for (fileSpec in spec.files) {
            val f = File(modelsRootDir, fileSpec.relativePath)
            val part = File(f.parentFile, "${f.name}.part")
            if (f.exists() && !f.delete()) allDeleted = false
            if (part.exists()) part.delete()
        }
        refreshInstallationStates()
        return allDeleted
    }

    private fun updateState(
        progress: ModelDownloadProgress,
        callback: ((ModelDownloadProgress) -> Unit)?
    ) {
        val next = _progressMap.value.toMutableMap()
        next[progress.modelId] = progress
        _progressMap.value = next
        callback?.invoke(progress)
    }
}
