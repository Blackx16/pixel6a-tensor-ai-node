package com.blackx16.tensorainode.ui

import android.Manifest
import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import com.blackx16.tensorainode.engine.ChatCompletionRequest
import com.blackx16.tensorainode.engine.ChatMessageInput
import com.blackx16.tensorainode.engine.UnifiedAiRouter
import com.blackx16.tensorainode.models.AiModelSpec
import com.blackx16.tensorainode.models.ModelCatalog
import com.blackx16.tensorainode.models.ModelDownloadProgress
import com.blackx16.tensorainode.server.AiHostForegroundService
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Ultra-minimal, zero-bloat native Android View frontend (<5 MB RAM overhead, 0% idle CPU).
 *
 * WHY: Extending base `android.app.Activity` and constructing native Android Views programmatically
 * avoids loading Jetpack Compose, AppCompat, Material Components, or WebView into memory. This saves
 * 30-60 MB of UI heap and GPU shader cache on the 6 GB Pixel 6a so the LLMs get maximum RAM.
 */
class MainActivity : Activity() {

    companion object {
        private const val REQ_PERMISSIONS = 1001
        private const val REQ_PICK_IMAGE = 1002
    }

    private val uiScope: CoroutineScope = MainScope()
    private var telemetryJob: Job? = null
    private var activeTestJob: Job? = null

    private var boundService: AiHostForegroundService? = null
    private var isBound = false

    private lateinit var telemetryTextView: TextView
    private lateinit var serverStatusTextView: TextView
    private lateinit var endpointTextView: TextView
    private lateinit var apiKeyInput: EditText
    private lateinit var toggleServerBtn: Button
    private lateinit var hfTokenInput: EditText
    private lateinit var customUrlInput: EditText
    private lateinit var modelsContainer: LinearLayout

    // On-Device Direct Test Console UI
    private lateinit var testModelSpinner: Spinner
    private lateinit var testModeSpinner: Spinner
    private lateinit var testPromptInput: EditText
    private lateinit var attachmentStatusText: TextView
    private lateinit var testOutputTextView: TextView
    private var attachedImageBytes: ByteArray? = null
    private var speechRecognizer: SpeechRecognizer? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as? AiHostForegroundService.LocalBinder ?: return
            boundService = binder.getService()
            isBound = true
            refreshServerUi()
            rebuildModelCards()
            startTelemetryLoop()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            boundService = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestAndroid17PermissionsIfNeeded()
        setContentView(buildMinimalRootView())

        val intent = Intent(this, AiHostForegroundService::class.java)
        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    override fun onDestroy() {
        telemetryJob?.cancel()
        activeTestJob?.cancel()
        speechRecognizer?.destroy()
        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }
        uiScope.cancel()
        super.onDestroy()
    }

    /**
     * WHY: Android 17 (API 37) requires runtime approval of `android.permission.ACCESS_LOCAL_NETWORK`
     * before an app can accept incoming TCP sockets on the local Wi-Fi subnet.
     */
    private fun requestAndroid17PermissionsIfNeeded() {
        val perms = mutableListOf(
            Manifest.permission.RECORD_AUDIO,
            "android.permission.ACCESS_LOCAL_NETWORK"
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            perms.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val missing = perms.filter {
            checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), REQ_PERMISSIONS)
        }
    }

    private fun buildMinimalRootView(): View {
        val scroll = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#0D1117"))
            isFillViewport = true
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = dp(12)
            setPadding(pad, pad, pad, pad)
        }
        scroll.addView(root)

        root.addView(
            makeSectionTitle("PIXEL 6A TENSOR GS101 • ANDROID 17 (API 37)")
        )
        telemetryTextView = makeMonoText(
            "SoC: Google Tensor GS101 (2x X1 + 2x A76 @ 0xF0) | GPU: Mali-G78 MP20\n" +
                "Build: DP11.261016.003 | Kernel: 6.1.172-android14-11 | Checking RAM..."
        )
        root.addView(makeCard(telemetryTextView))

        // 1. Local Wi-Fi OpenAI Host Section
        root.addView(makeSectionTitle("1. LOCAL WI-FI OPENAI API HOST"))
        val serverCardLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        serverStatusTextView = makeMonoText("Server Status: STOPPED")
        endpointTextView = makeMonoText("Endpoint: http://0.0.0.0:8080/v1")
        serverCardLayout.addView(serverStatusTextView)
        serverCardLayout.addView(endpointTextView)

        apiKeyInput = makeInput("OpenAI Bearer API Key", "sk-pixel6a-tensor")
        serverCardLayout.addView(apiKeyInput)

        val serverBtnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        toggleServerBtn = makeButton("START WI-FI SERVER", "#238636") {
            toggleOpenAiServer()
        }
        val copyCurlBtn = makeButton("COPY CURL", "#1F6FEB") {
            copySampleCurlToClipboard()
        }
        serverBtnRow.addView(
            toggleServerBtn,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { rightMargin = dp(6) }
        )
        serverBtnRow.addView(
            copyCurlBtn,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        )
        serverCardLayout.addView(serverBtnRow)
        root.addView(makeCard(serverCardLayout))

        // 2. Post-Install Web Model Downloader Section
        root.addView(makeSectionTitle("2. WEB MODEL DOWNLOADER (LITERT & ONNX)"))
        val dlWrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        hfTokenInput = makeInput("Optional Hugging Face Token (hf_...) for gated models", "")
        dlWrap.addView(hfTokenInput)

        val customRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        customUrlInput = makeInput("Custom .litertlm / .onnx HTTPS URL", "").apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        val addUrlBtn = makeButton("ADD URL", "#30363D") {
            val url = customUrlInput.text.toString().trim()
            val svc = boundService ?: return@makeButton
            if (url.startsWith("http")) {
                svc.downloadManager.registerCustomModelUrl(url)
                customUrlInput.setText("")
                rebuildModelCards()
            } else {
                Toast.makeText(this, "Enter a valid https:// model URL", Toast.LENGTH_SHORT).show()
            }
        }
        customRow.addView(customUrlInput)
        customRow.addView(addUrlBtn)
        dlWrap.addView(customRow)

        modelsContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        dlWrap.addView(modelsContainer)
        root.addView(makeCard(dlWrap))

        // 3. On-Device Direct Test Section
        root.addView(makeSectionTitle("3. ON-DEVICE DIRECT TEST CONSOLE"))
        val testLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }

        testModelSpinner = Spinner(this).apply {
            setBackgroundColor(Color.parseColor("#21262D"))
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf(
                    "${ModelCatalog.PRIMARY_LITERT_LLM_ID} (LiteRT Multimodal)",
                    "${ModelCatalog.PRIMARY_ONNX_LLM_ID} (ONNX INT4)",
                    "${ModelCatalog.ONNX_EMBED_ID} (ONNX 384-D Embeddings)",
                    "${ModelCatalog.LITERT_EMBED_ID} (LiteRT 100-D Embeddings)",
                    "${ModelCatalog.ONNX_WHISPER_ID} (ONNX Whisper Voice ASR)"
                )
            )
        }
        testLayout.addView(testModelSpinner)

        testModeSpinner = Spinner(this).apply {
            setBackgroundColor(Color.parseColor("#21262D"))
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf(
                    "Mode: Text / Multimodal Chat",
                    "Mode: Embeddings Vector + Cosine Similarity",
                    "Mode: Voice TTS Speech Synthesis"
                )
            )
        }
        testLayout.addView(testModeSpinner)

        testPromptInput = makeInput(
            "Enter prompt (or two lines for Embedding similarity test)...",
            "Explain how Google Tensor GS101 executes INT4 matrix multiplication."
        ).apply {
            minLines = 2
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        }
        testLayout.addView(testPromptInput)

        attachmentStatusText = makeMonoText("Attachments: None (Text-only)")
        testLayout.addView(attachmentStatusText)

        val modalityBtnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        val pickImgBtn = makeButton("+ IMAGE", "#30363D") {
            pickImageFromGallery()
        }
        val voiceMicBtn = makeButton("MIC VOICE", "#30363D") {
            startVoiceDictation()
        }
        val runTestBtn = makeButton("RUN ON PHONE", "#238636") {
            runDirectOnDeviceTest()
        }
        modalityBtnRow.addView(
            pickImgBtn,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { rightMargin = dp(4) }
        )
        modalityBtnRow.addView(
            voiceMicBtn,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { rightMargin = dp(4) }
        )
        modalityBtnRow.addView(
            runTestBtn,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1.3f)
        )
        testLayout.addView(modalityBtnRow)

        testOutputTextView = makeMonoText("Output will stream here directly from your Pixel 6a...")
        testLayout.addView(testOutputTextView)

        root.addView(makeCard(testLayout))
        return scroll
    }

    private fun startTelemetryLoop() {
        telemetryJob?.cancel()
        telemetryJob = uiScope.launch {
            while (isActive) {
                val svc = boundService
                if (svc != null) {
                    val mem = svc.optimizer.readMemoryTelemetry()
                    val ip = svc.optimizer.getLocalWifiIpAddress()
                    val srvRunning = svc.httpServer?.isRunning == true
                    telemetryTextView.text =
                        "SoC: Tensor GS101 (4x Big Cores 0xF0 Pinned) | Page: ${mem.pageSizeBytes} B\n" +
                        "RAM Free: ${mem.availableDeviceRamMb}/${mem.totalDeviceRamMb} MB | " +
                        "AnonRSS: ${mem.rssAnonMb} MB | mmap(RssFile): ${mem.rssFileMmapMb} MB"
                    endpointTextView.text = "Wi-Fi Endpoint: http://$ip:8080/v1"
                    serverStatusTextView.text = if (srvRunning) {
                        "Server Status: ACTIVE ON LOCAL WI-FI (Port 8080)"
                    } else {
                        "Server Status: STOPPED"
                    }
                    toggleServerBtn.text = if (srvRunning) "STOP WI-FI SERVER" else "START WI-FI SERVER"
                }
                delay(2000L)
            }
        }
    }

    private fun toggleOpenAiServer() {
        val svc = boundService ?: return
        val running = svc.httpServer?.isRunning == true
        if (running) {
            val stopIntent = Intent(this, AiHostForegroundService::class.java).apply {
                action = AiHostForegroundService.ACTION_STOP_SERVER
            }
            startService(stopIntent)
            svc.stopOpenAiServer()
        } else {
            val key = apiKeyInput.text.toString().trim().ifBlank { "sk-pixel6a-tensor" }
            val startIntent = Intent(this, AiHostForegroundService::class.java).apply {
                action = AiHostForegroundService.ACTION_START_SERVER
                putExtra(AiHostForegroundService.EXTRA_API_KEY, key)
                putExtra(AiHostForegroundService.EXTRA_PORT, 8080)
            }
            startForegroundService(startIntent)
        }
        refreshServerUi()
    }

    private fun refreshServerUi() {
        val svc = boundService ?: return
        val ip = svc.optimizer.getLocalWifiIpAddress()
        val running = svc.httpServer?.isRunning == true
        endpointTextView.text = "Wi-Fi Endpoint: http://$ip:8080/v1"
        serverStatusTextView.text = if (running) {
            "Server Status: ACTIVE ON LOCAL WI-FI (Port 8080)"
        } else {
            "Server Status: STOPPED"
        }
        toggleServerBtn.text = if (running) "STOP WI-FI SERVER" else "START WI-FI SERVER"
    }

    private fun copySampleCurlToClipboard() {
        val svc = boundService ?: return
        val ip = svc.optimizer.getLocalWifiIpAddress()
        val key = apiKeyInput.text.toString().trim().ifBlank { "sk-pixel6a-tensor" }
        val curl = "curl http://$ip:8080/v1/chat/completions -H \"Authorization: Bearer $key\" -H \"Content-Type: application/json\" -d '{\"model\":\"${ModelCatalog.PRIMARY_LITERT_LLM_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello from LAN!\"}],\"stream\":true}'"
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
        cm?.setPrimaryClip(ClipData.newPlainText("OpenAI cURL", curl))
        Toast.makeText(this, "Copied OpenAI cURL command to clipboard", Toast.LENGTH_SHORT).show()
    }

    private fun rebuildModelCards() {
        val svc = boundService ?: return
        modelsContainer.removeAllViews()
        val allModels = svc.downloadManager.getAllModels()

        for (spec in allModels) {
            val row = buildSingleModelRow(spec)
            modelsContainer.addView(row)
        }
    }

    private fun buildSingleModelRow(spec: AiModelSpec): View {
        val svc = boundService!!
        val installed = svc.downloadManager.isModelInstalled(spec)

        val wrap = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = dp(8)
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(Color.parseColor("#161B22"))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(6) }
        }

        val titleView = TextView(this).apply {
            text = "${spec.title} [${spec.totalDownloadMb} MB • ~${spec.estimatedRamMb} MB RAM]"
            setTextColor(Color.parseColor("#F0F6FC"))
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
        }
        val descView = TextView(this).apply {
            text = "${spec.runtime.displayName} • ${spec.quantization}\n${spec.description}"
            setTextColor(Color.parseColor("#8B949E"))
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
        }
        val progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 100
            progress = if (installed) 100 else 0
        }
        val statusLabel = makeMonoText(
            if (installed) "Status: INSTALLED & READY ON UFS 3.1" else "Status: Not downloaded (0 B on APK)"
        )

        val btnRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        val actionBtn = makeButton(
            if (installed) "RE-VERIFY / READY" else "DOWNLOAD OVER WEB",
            if (installed) "#1F6FEB" else "#238636"
        ) {
            val token = hfTokenInput.text.toString().trim()
            uiScope.launch {
                svc.downloadManager.downloadModel(spec, token) { prog: ModelDownloadProgress ->
                    runOnUiThread {
                        progressBar.progress = prog.percent
                        statusLabel.text = if (prog.error != null) {
                            "Error: ${prog.error}"
                        } else if (prog.isInstalled) {
                            "Status: INSTALLED (${prog.bytesDownloaded / (1024 * 1024)} MB)"
                        } else {
                            "Downloading: ${prog.percent}% (${prog.bytesDownloaded / (1024 * 1024)} / ${spec.totalDownloadMb} MB)"
                        }
                    }
                }
                rebuildModelCards()
            }
        }

        val deleteBtn = makeButton("DELETE", "#DA3633") {
            svc.downloadManager.deleteModel(spec)
            rebuildModelCards()
        }

        btnRow.addView(
            actionBtn,
            LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { rightMargin = dp(6) }
        )
        btnRow.addView(
            deleteBtn,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        )

        wrap.addView(titleView)
        wrap.addView(descView)
        wrap.addView(progressBar)
        wrap.addView(statusLabel)
        wrap.addView(btnRow)
        return wrap
    }

    private fun pickImageFromGallery() {
        val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "image/*"
        }
        startActivityForResult(Intent.createChooser(intent, "Select Image for Vision Test"), REQ_PICK_IMAGE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_PICK_IMAGE && resultCode == RESULT_OK) {
            val uri: Uri = data?.data ?: return
            try {
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes != null) {
                    attachedImageBytes = bytes
                    attachmentStatusText.text = "Attachment: Image loaded (${bytes.size / 1024} KB) for Vision test"
                }
            } catch (e: Throwable) {
                attachmentStatusText.text = "Attachment error: ${e.message}"
            }
        }
    }

    private fun startVoiceDictation() {
        if (!SpeechRecognizer.isRecognitionAvailable(this)) {
            Toast.makeText(this, "SpeechRecognizer unavailable on this build", Toast.LENGTH_SHORT).show()
            return
        }
        if (speechRecognizer == null) {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        }
        val recognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
        }
        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                attachmentStatusText.text = "Voice: Listening on Pixel 6a mic..."
            }
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {
                attachmentStatusText.text = "Voice: Processing speech..."
            }
            override fun onError(error: Int) {
                attachmentStatusText.text = "Voice error code: $error"
            }
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val transcript = matches?.firstOrNull().orEmpty()
                if (transcript.isNotBlank()) {
                    testPromptInput.setText(transcript)
                    attachmentStatusText.text = "Voice captured: \"$transcript\""
                }
            }
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        speechRecognizer?.startListening(recognizerIntent)
    }

    /**
     * WHY: Runs inference directly against the in-process `UnifiedAiRouter` without going over a network
     * socket so the user can verify LiteRT and ONNX models locally on the Pixel 6a at any time.
     */
    private fun runDirectOnDeviceTest() {
        val svc = boundService ?: return
        val selectedModelLine = testModelSpinner.selectedItem?.toString().orEmpty()
        val selectedModelId = selectedModelLine.substringBefore(" ").trim()
        val modeIdx = testModeSpinner.selectedItemPosition
        val prompt = testPromptInput.text.toString().trim()

        activeTestJob?.cancel()
        activeTestJob = uiScope.launch(Dispatchers.Main) {
            when (modeIdx) {
                0 -> {
                    testOutputTextView.text = "[Running on $selectedModelId (Tensor GS101 0xF0)...]\n\n"
                    val imgList = attachedImageBytes?.let { listOf(it) } ?: emptyList()
                    val req = ChatCompletionRequest(
                        model = selectedModelId,
                        messages = listOf(
                            ChatMessageInput(
                                role = "user",
                                text = prompt,
                                imageBytesList = imgList
                            )
                        ),
                        stream = true
                    )
                    val sb = StringBuilder()
                    var statsFooter = ""
                    svc.router.streamChatCompletion(req).collect { chunk ->
                        sb.append(chunk.deltaText)
                        statsFooter = "\n\n[Metrics: ${chunk.completionTokens} tokens | " +
                            "${"%.1f".format(chunk.tokensPerSecond)} tok/s]"
                        testOutputTextView.text = sb.toString() + statsFooter
                    }
                }
                1 -> {
                    val lines = prompt.lines().map { it.trim() }.filter { it.isNotEmpty() }
                    val inputs = if (lines.size >= 2) lines.take(2) else listOf(prompt, "Google Tensor mobile AI processor")
                    testOutputTextView.text = "Computing embeddings on $selectedModelId..."
                    val emb = svc.router.generateEmbeddings(selectedModelId, inputs)
                    val v0 = emb.embeddings[0]
                    val v1 = emb.embeddings.getOrElse(1) { v0 }
                    val sim = UnifiedAiRouter.cosineSimilarity(v0, v1)
                    val preview = v0.take(6).joinToString(", ") { "%.4f".format(it) }
                    testOutputTextView.text =
                        "Model: ${emb.model}\n" +
                        "Dimension: ${v0.size}-D (L2 Normalized) | Latency: ${emb.latencyMs} ms\n" +
                        "Vector[0..5]: [$preview, ...]\n" +
                        "Cosine Similarity(\"${inputs[0]}\", \"${inputs[1]}\") = ${"%.4f".format(sim)}"
                }
                2 -> {
                    testOutputTextView.text = "Synthesizing WAV speech via Pixel Tensor TTS..."
                    val wavBytes = svc.router.synthesizeSpeech(prompt)
                    testOutputTextView.text =
                        "Synthesized ${wavBytes.size / 1024} KB RIFF WAVE audio (16 kHz Mono PCM) ready for /v1/audio/speech."
                }
            }
        }
    }

    private fun makeSectionTitle(title: String): TextView =
        TextView(this).apply {
            text = title
            setTextColor(Color.parseColor("#58A6FF"))
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setPadding(0, dp(10), 0, dp(4))
        }

    private fun makeMonoText(initial: String): TextView =
        TextView(this).apply {
            text = initial
            setTextColor(Color.parseColor("#C9D1D9"))
            typeface = Typeface.MONOSPACE
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            setPadding(0, dp(3), 0, dp(3))
        }

    private fun makeInput(hintText: String, defaultText: String): EditText =
        EditText(this).apply {
            hint = hintText
            setText(defaultText)
            setTextColor(Color.parseColor("#F0F6FC"))
            setHintTextColor(Color.parseColor("#6E7681"))
            setBackgroundColor(Color.parseColor("#0D1117"))
            typeface = Typeface.MONOSPACE
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            val pad = dp(8)
            setPadding(pad, pad, pad, pad)
        }

    private fun makeButton(label: String, hexBg: String, onClick: () -> Unit): Button =
        Button(this).apply {
            text = label
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor(hexBg))
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
            gravity = Gravity.CENTER
            setOnClickListener { onClick() }
        }

    private fun makeCard(content: View): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val pad = dp(10)
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(Color.parseColor("#161B22"))
            addView(content)
        }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()
}
