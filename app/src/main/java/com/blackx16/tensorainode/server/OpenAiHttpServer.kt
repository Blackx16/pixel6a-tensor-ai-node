package com.blackx16.tensorainode.server

import android.util.Base64
import com.blackx16.tensorainode.engine.ChatCompletionRequest
import com.blackx16.tensorainode.engine.ChatMessageInput
import com.blackx16.tensorainode.engine.UnifiedAiRouter
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.security.MessageDigest
import java.util.UUID

/**
 * Ultra-lightweight (<1 MB RAM overhead), zero-dependency OpenAI-compatible HTTP/1.1 Server
 * designed to run on the Pixel 6a's local Wi-Fi interface (`0.0.0.0:<port>`).
 *
 * WHY: Instead of pulling in heavy Netty/Jetty web server dependencies that consume 15-30 MB of
 * RAM and spawn dozens of threads, this custom coroutine socket server runs on the 2-thread
 * Cortex-A55 efficiency core pool (`0x0F` affinity) with near-zero heap overhead.
 */
class OpenAiHttpServer(
    private val router: UnifiedAiRouter,
    @Volatile var apiKey: String = "sk-pixel6a-tensor",
    val port: Int = 8080,
    private val ioDispatcher: CoroutineDispatcher = Dispatchers.IO
) {
    private var serverSocket: ServerSocket? = null
    private var serverScope: CoroutineScope? = null
    private var acceptJob: Job? = null

    val isRunning: Boolean
        get() = serverSocket?.isBound == true && serverSocket?.isClosed == false

    @Synchronized
    fun start() {
        if (isRunning) return
        val scope = CoroutineScope(SupervisorJob() + ioDispatcher)
        val ss = ServerSocket()
        ss.reuseAddress = true
        ss.bind(InetSocketAddress("0.0.0.0", port), 32)
        serverSocket = ss
        serverScope = scope

        acceptJob = scope.launch {
            while (isActive && !ss.isClosed) {
                try {
                    val client = ss.accept()
                    client.tcpNoDelay = true
                    client.soTimeout = 120_000
                    launch {
                        handleClient(client)
                    }
                } catch (_: Throwable) {
                    if (ss.isClosed) break
                }
            }
        }
    }

    @Synchronized
    fun stop() {
        try {
            serverSocket?.close()
        } catch (_: Throwable) {
        }
        serverSocket = null
        acceptJob?.cancel()
        serverScope?.cancel()
        serverScope = null
    }

    private suspend fun handleClient(socket: Socket) {
        socket.use { sock ->
            val input = sock.getInputStream()
            val output = BufferedOutputStream(sock.getOutputStream(), 16 * 1024)

            val request = parseHttpRequest(input) ?: return
            if (request.method.equals("OPTIONS", ignoreCase = true)) {
                writeJsonResponse(output, 200, "OK", "{\"status\":\"ok\"}")
                return
            }

            if (request.path == "/health") {
                val mem = router.optimizer.readMemoryTelemetry()
                val healthJson = JSONObject()
                    .put("status", "ok")
                    .put("device", "Google Pixel 6a (bluejay)")
                    .put("soc", "Google Tensor GS101")
                    .put("os", "Android 17 (API 37 / DP11.261016.003)")
                    .put("rss_anon_mb", mem.rssAnonMb)
                    .put("rss_file_mmap_mb", mem.rssFileMmapMb)
                    .toString()
                writeJsonResponse(output, 200, "OK", healthJson)
                return
            }

            // WHY: Enforce standard OpenAI `Authorization: Bearer <api_key>` format for all `/v1/*` routes.
            if (!isAuthorized(request.headers["authorization"], apiKey)) {
                val err = buildOpenAiError(
                    "Invalid or missing Bearer API key. Pass header 'Authorization: Bearer $apiKey'.",
                    "invalid_api_key",
                    "401"
                )
                writeJsonResponse(output, 401, "Unauthorized", err)
                return
            }

            when {
                request.method == "GET" && request.path == "/v1/models" -> {
                    handleListModels(output)
                }
                request.method == "POST" && request.path == "/v1/chat/completions" -> {
                    handleChatCompletions(request, output)
                }
                request.method == "POST" && request.path == "/v1/embeddings" -> {
                    handleEmbeddings(request, output)
                }
                request.method == "POST" && request.path == "/v1/audio/transcriptions" -> {
                    handleAudioTranscriptions(request, output)
                }
                request.method == "POST" && request.path == "/v1/audio/speech" -> {
                    handleAudioSpeech(request, output)
                }
                else -> {
                    val notFound = buildOpenAiError(
                        "Unknown endpoint ${request.method} ${request.path}",
                        "invalid_request_error",
                        "404"
                    )
                    writeJsonResponse(output, 404, "Not Found", notFound)
                }
            }
        }
    }

    private fun handleListModels(output: BufferedOutputStream) {
        val dataArr = JSONArray()
        val createdTs = 1762490000L
        for (spec in router.downloadManager.getAllModels()) {
            val installed = router.downloadManager.isModelInstalled(spec)
            val modelObj = JSONObject()
                .put("id", spec.id)
                .put("object", "model")
                .put("created", createdTs)
                .put("owned_by", spec.runtime.name.lowercase())
                .put("installed", installed)
                .put("quantization", spec.quantization)
                .put("estimated_ram_mb", spec.estimatedRamMb)
                .put("capabilities", JSONArray(spec.capabilities.map { it.name.lowercase() }))
            dataArr.put(modelObj)
        }
        val resp = JSONObject()
            .put("object", "list")
            .put("data", dataArr)
            .toString()
        writeJsonResponse(output, 200, "OK", resp)
    }

    private suspend fun handleChatCompletions(
        httpReq: RawHttpRequest,
        output: BufferedOutputStream
    ) {
        val bodyStr = httpReq.body.toString(Charsets.UTF_8)
        val json = try {
            JSONObject(bodyStr)
        } catch (e: Throwable) {
            writeJsonResponse(
                output,
                400,
                "Bad Request",
                buildOpenAiError("Invalid JSON body: ${e.message}", "invalid_request_error", "400")
            )
            return
        }

        val chatReq = parseChatCompletionRequest(json)
        val completionId = "chatcmpl-${UUID.randomUUID().toString().replace("-", "").take(16)}"
        val created = System.currentTimeMillis() / 1000L
        val resolvedSpec = router.resolveChatModel(chatReq.model)

        if (chatReq.stream) {
            // WHY: Flush each token immediately as an OpenAI Server-Sent Event (`data: {...}\n\n`)
            // so external LAN clients receive real-time token streaming over Wi-Fi.
            val sseHeaders = buildString {
                append("HTTP/1.1 200 OK\r\n")
                append("Content-Type: text/event-stream; charset=utf-8\r\n")
                append("Cache-Control: no-cache\r\n")
                append("Connection: keep-alive\r\n")
                append("Access-Control-Allow-Origin: *\r\n\r\n")
            }
            output.write(sseHeaders.toByteArray(Charsets.UTF_8))
            output.flush()

            router.streamChatCompletion(chatReq).collect { chunk ->
                val choiceObj = JSONObject().put("index", 0)
                if (chunk.isFinished) {
                    choiceObj.put("delta", JSONObject())
                    choiceObj.put("finish_reason", "stop")
                } else {
                    choiceObj.put(
                        "delta",
                        JSONObject()
                            .put("role", "assistant")
                            .put("content", chunk.deltaText)
                    )
                    choiceObj.put("finish_reason", JSONObject.NULL)
                }

                val eventJson = JSONObject()
                    .put("id", completionId)
                    .put("object", "chat.completion.chunk")
                    .put("created", created)
                    .put("model", resolvedSpec.id)
                    .put("choices", JSONArray().put(choiceObj))

                val sseFrame = "data: $eventJson\n\n"
                output.write(sseFrame.toByteArray(Charsets.UTF_8))
                output.flush()
            }

            output.write("data: [DONE]\n\n".toByteArray(Charsets.UTF_8))
            output.flush()
        } else {
            val fullText = StringBuilder()
            var promptTok = 0
            var compTok = 0
            var tokPerSec = 0f

            router.streamChatCompletion(chatReq).collect { chunk ->
                fullText.append(chunk.deltaText)
                promptTok = chunk.promptTokens
                compTok = chunk.completionTokens
                tokPerSec = chunk.tokensPerSecond
            }

            val messageObj = JSONObject()
                .put("role", "assistant")
                .put("content", fullText.toString())

            val choiceObj = JSONObject()
                .put("index", 0)
                .put("message", messageObj)
                .put("finish_reason", "stop")

            val usageObj = JSONObject()
                .put("prompt_tokens", promptTok)
                .put("completion_tokens", compTok)
                .put("total_tokens", promptTok + compTok)
                .put("tokens_per_second", tokPerSec)

            val responseJson = JSONObject()
                .put("id", completionId)
                .put("object", "chat.completion")
                .put("created", created)
                .put("model", resolvedSpec.id)
                .put("choices", JSONArray().put(choiceObj))
                .put("usage", usageObj)
                .toString()

            writeJsonResponse(output, 200, "OK", responseJson)
        }
    }

    private suspend fun handleEmbeddings(
        httpReq: RawHttpRequest,
        output: BufferedOutputStream
    ) {
        val json = try {
            JSONObject(httpReq.body.toString(Charsets.UTF_8))
        } catch (e: Throwable) {
            writeJsonResponse(
                output,
                400,
                "Bad Request",
                buildOpenAiError("Invalid JSON body: ${e.message}", "invalid_request_error", "400")
            )
            return
        }

        val modelId = json.optString("model", "")
        val inputs = mutableListOf<String>()
        val rawInput = json.opt("input")
        when (rawInput) {
            is JSONArray -> {
                for (i in 0 until rawInput.length()) {
                    inputs.add(rawInput.optString(i, ""))
                }
            }
            is String -> inputs.add(rawInput)
            else -> inputs.add(rawInput?.toString().orEmpty())
        }

        val embData = router.generateEmbeddings(modelId, inputs)
        val dataArr = JSONArray()
        embData.embeddings.forEachIndexed { idx, vec ->
            val floatArr = JSONArray()
            for (v in vec) floatArr.put(v.toDouble())
            val item = JSONObject()
                .put("object", "embedding")
                .put("index", idx)
                .put("embedding", floatArr)
            dataArr.put(item)
        }

        val resp = JSONObject()
            .put("object", "list")
            .put("data", dataArr)
            .put("model", embData.model)
            .put(
                "usage",
                JSONObject()
                    .put("prompt_tokens", embData.promptTokens)
                    .put("total_tokens", embData.promptTokens)
            )
            .toString()

        writeJsonResponse(output, 200, "OK", resp)
    }

    private suspend fun handleAudioTranscriptions(
        httpReq: RawHttpRequest,
        output: BufferedOutputStream
    ) {
        val contentType = httpReq.headers["content-type"].orEmpty()
        val audioBytes = if (contentType.contains("multipart/form-data", ignoreCase = true)) {
            extractMultipartFileBytes(httpReq.body, contentType)
        } else {
            try {
                val json = JSONObject(httpReq.body.toString(Charsets.UTF_8))
                val b64 = json.optString("audio", "")
                if (b64.isNotBlank()) decodeBase64Portable(b64) else httpReq.body
            } catch (_: Throwable) {
                httpReq.body
            }
        }

        val result = router.transcribeAudio(audioBytes)
        val resp = JSONObject()
            .put("text", result.text)
            .put("model", result.model)
            .put("duration_ms", result.durationMs)
            .toString()
        writeJsonResponse(output, 200, "OK", resp)
    }

    private suspend fun handleAudioSpeech(
        httpReq: RawHttpRequest,
        output: BufferedOutputStream
    ) {
        val json = try {
            JSONObject(httpReq.body.toString(Charsets.UTF_8))
        } catch (e: Throwable) {
            writeJsonResponse(
                output,
                400,
                "Bad Request",
                buildOpenAiError("Invalid JSON body: ${e.message}", "invalid_request_error", "400")
            )
            return
        }

        val inputText = json.optString("input", "Hello from Pixel 6a Tensor AI server.")
        val speed = json.optDouble("speed", 1.0).toFloat()
        val wavBytes = router.synthesizeSpeech(inputText, speed)

        val headers = buildString {
            append("HTTP/1.1 200 OK\r\n")
            append("Content-Type: audio/wav\r\n")
            append("Content-Length: ${wavBytes.size}\r\n")
            append("Access-Control-Allow-Origin: *\r\n")
            append("Connection: close\r\n\r\n")
        }
        output.write(headers.toByteArray(Charsets.UTF_8))
        output.write(wavBytes)
        output.flush()
    }

    companion object {
        data class RawHttpRequest(
            val method: String,
            val path: String,
            val headers: Map<String, String>,
            val body: ByteArray
        )

        /**
         * WHY: Constant-time comparison (`MessageDigest.isEqual`) prevents timing side-channel
         * leaks when verifying Bearer API keys over local Wi-Fi.
         */
        fun isAuthorized(authHeader: String?, expectedApiKey: String): Boolean {
            if (expectedApiKey.isBlank()) return true
            if (authHeader.isNullOrBlank()) return false
            val prefix = "Bearer "
            if (!authHeader.startsWith(prefix, ignoreCase = true)) return false
            val provided = authHeader.substring(prefix.length).trim()
            return MessageDigest.isEqual(
                provided.toByteArray(Charsets.UTF_8),
                expectedApiKey.trim().toByteArray(Charsets.UTF_8)
            )
        }

        fun parseHttpRequest(input: InputStream): RawHttpRequest? {
            val requestLine = readAsciiLine(input) ?: return null
            if (requestLine.isBlank()) return null
            val parts = requestLine.split(" ")
            if (parts.size < 2) return null
            val method = parts[0].trim().uppercase()
            val fullPath = parts[1].trim()
            val path = fullPath.substringBefore("?")

            val headers = HashMap<String, String>()
            while (true) {
                val line = readAsciiLine(input) ?: break
                if (line.isEmpty()) break
                val colon = line.indexOf(':')
                if (colon > 0) {
                    val key = line.substring(0, colon).trim().lowercase()
                    val value = line.substring(colon + 1).trim()
                    headers[key] = value
                }
            }

            val contentLength = headers["content-length"]?.toIntOrNull() ?: 0
            val body = if (contentLength > 0) {
                val buf = ByteArray(contentLength)
                var readTotal = 0
                while (readTotal < contentLength) {
                    val r = input.read(buf, readTotal, contentLength - readTotal)
                    if (r == -1) break
                    readTotal += r
                }
                buf
            } else {
                ByteArray(0)
            }

            return RawHttpRequest(method, path, headers, body)
        }

        private fun readAsciiLine(input: InputStream): String? {
            val baos = ByteArrayOutputStream(128)
            var prevCr = false
            while (true) {
                val b = input.read()
                if (b == -1) {
                    return if (baos.size() == 0) null else baos.toString("UTF-8")
                }
                if (b == '\n'.code) {
                    val arr = baos.toByteArray()
                    val len = if (prevCr && arr.isNotEmpty()) arr.size - 1 else arr.size
                    return String(arr, 0, len, Charsets.UTF_8)
                }
                prevCr = (b == '\r'.code)
                baos.write(b)
            }
        }

        fun parseChatCompletionRequest(json: JSONObject): ChatCompletionRequest {
            val model = json.optString("model", "")
            val temperature = json.optDouble("temperature", 0.7).toFloat()
            val topP = json.optDouble("top_p", 0.9).toFloat()
            val maxTokens = json.optInt("max_tokens", 512)
            val stream = json.optBoolean("stream", false)

            val messagesArr = json.optJSONArray("messages") ?: JSONArray()
            val parsedMessages = ArrayList<ChatMessageInput>(messagesArr.length())

            for (i in 0 until messagesArr.length()) {
                val mObj = messagesArr.optJSONObject(i) ?: continue
                val role = mObj.optString("role", "user")
                val rawContent = mObj.opt("content")

                val textParts = StringBuilder()
                val images = mutableListOf<ByteArray>()
                val audios = mutableListOf<ByteArray>()

                when (rawContent) {
                    is String -> textParts.append(rawContent)
                    is JSONArray -> {
                        for (j in 0 until rawContent.length()) {
                            val block = rawContent.optJSONObject(j) ?: continue
                            when (block.optString("type")) {
                                "text" -> textParts.append(block.optString("text", "")).append(" ")
                                "image_url" -> {
                                    val urlObj = block.optJSONObject("image_url")
                                    val urlStr = urlObj?.optString("url") ?: block.optString("image_url", "")
                                    if (urlStr.startsWith("data:image", ignoreCase = true)) {
                                        val b64 = urlStr.substringAfter("base64,", "")
                                        if (b64.isNotEmpty()) {
                                            images.add(decodeBase64Portable(b64))
                                        }
                                    }
                                }
                                "input_audio" -> {
                                    val audObj = block.optJSONObject("input_audio")
                                    val b64 = audObj?.optString("data").orEmpty()
                                    if (b64.isNotEmpty()) {
                                        audios.add(decodeBase64Portable(b64))
                                    }
                                }
                            }
                        }
                    }
                }

                parsedMessages.add(
                    ChatMessageInput(
                        role = role,
                        text = textParts.toString().trim(),
                        imageBytesList = images,
                        audioBytesList = audios
                    )
                )
            }

            return ChatCompletionRequest(
                model = model,
                messages = parsedMessages,
                temperature = temperature,
                topP = topP,
                maxTokens = maxTokens,
                stream = stream
            )
        }

        fun extractMultipartFileBytes(body: ByteArray, contentTypeHeader: String): ByteArray {
            val boundary = contentTypeHeader.substringAfter("boundary=", "").trim().trim('"')
            if (boundary.isEmpty()) return body
            val bodyStr = String(body, Charsets.ISO_8859_1)
            val headerEndIdx = bodyStr.indexOf("\r\n\r\n")
            if (headerEndIdx < 0) return body
            val dataStart = headerEndIdx + 4
            val closingMarker = "\r\n--$boundary"
            val dataEnd = bodyStr.indexOf(closingMarker, dataStart).takeIf { it >= dataStart } ?: body.size
            return body.copyOfRange(dataStart, dataEnd)
        }

        fun decodeBase64Portable(b64: String): ByteArray {
            return try {
                java.util.Base64.getDecoder().decode(b64.trim())
            } catch (_: Throwable) {
                try {
                    Base64.decode(b64.trim(), Base64.DEFAULT)
                } catch (_: Throwable) {
                    ByteArray(0)
                }
            }
        }

        private fun buildOpenAiError(message: String, type: String, code: String): String =
            JSONObject()
                .put(
                    "error",
                    JSONObject()
                        .put("message", message)
                        .put("type", type)
                        .put("code", code)
                )
                .toString()

        private fun writeJsonResponse(
            output: BufferedOutputStream,
            statusCode: Int,
            statusText: String,
            jsonBody: String
        ) {
            val payload = jsonBody.toByteArray(Charsets.UTF_8)
            val headers = buildString {
                append("HTTP/1.1 $statusCode $statusText\r\n")
                append("Content-Type: application/json; charset=utf-8\r\n")
                append("Content-Length: ${payload.size}\r\n")
                append("Access-Control-Allow-Origin: *\r\n")
                append("Access-Control-Allow-Headers: Authorization, Content-Type\r\n")
                append("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
                append("Connection: close\r\n\r\n")
            }
            output.write(headers.toByteArray(Charsets.UTF_8))
            output.write(payload)
            output.flush()
        }
    }
}
