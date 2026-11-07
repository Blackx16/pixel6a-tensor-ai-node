package com.blackx16.tensorainode.engine

import android.content.Context
import android.graphics.BitmapFactory
import com.blackx16.tensorainode.hardware.TensorHardwareOptimizer
import com.blackx16.tensorainode.models.AiModelSpec
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.genai.llminference.GraphOptions
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.sqrt

/**
 * Google LiteRT ("Lightartie") Inference Engine for Pixel 6a (Tensor GS101 + Mali-G78 MP20 GPU).
 */
class LiteRtInferenceEngine(
    private val context: Context,
    private val optimizer: TensorHardwareOptimizer
) {
    private var activeModelId: String? = null
    private var litertLmEngine: Engine? = null
    private var mediaPipeLlm: LlmInference? = null
    private var textEmbedder: TextEmbedder? = null
    private var activeEmbedderId: String? = null

    @Synchronized
    fun unloadLlm() {
        try {
            litertLmEngine?.close()
        } catch (_: Throwable) {
        }
        litertLmEngine = null

        try {
            mediaPipeLlm?.close()
        } catch (_: Throwable) {
        }
        mediaPipeLlm = null
        activeModelId = null
    }

    @Synchronized
    fun unloadEmbedder() {
        try {
            textEmbedder?.close()
        } catch (_: Throwable) {
        }
        textEmbedder = null
        activeEmbedderId = null
    }

    /**
     * WHY: Pre-maps the model file into the Linux page cache (RssFile) before initializing
     * either LiteRT-LM (`Engine` for `.litertlm` files) or MediaPipe (`LlmInference` for `.task` files),
     * and caps `maxNumTokens = 2048` so the KV cache never breaches the Pixel 6a's 6 GB RAM ceiling.
     */
    private fun ensureLlmLoaded(spec: AiModelSpec, modelFile: File) {
        if (activeModelId == spec.id && (litertLmEngine != null || mediaPipeLlm != null)) {
            return
        }
        unloadLlm()
        optimizer.mmapModelWeights(modelFile)

        if (modelFile.name.endsWith(".litertlm", ignoreCase = true)) {
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = Backend.GPU(),
                visionBackend = Backend.GPU(),
                audioBackend = Backend.CPU(),
                maxNumTokens = 2048
            )
            val engine = Engine(config)
            engine.initialize()
            litertLmEngine = engine
        } else {
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(2048)
                .setMaxNumImages(4)
                .setPreferredBackend(LlmInference.Backend.GPU)
                .build()
            mediaPipeLlm = LlmInference.createFromOptions(context, options)
        }
        activeModelId = spec.id
    }

    fun streamChat(
        spec: AiModelSpec,
        modelFile: File,
        request: ChatCompletionRequest
    ): Flow<TokenStreamChunk> = callbackFlow {
        val startNs = System.nanoTime()
        var tokenCount = 0

        try {
            ensureLlmLoaded(spec, modelFile)
            val promptText = formatChatPrompt(request.messages)
            val promptTokensEst = (promptText.length / 4).coerceAtLeast(1)

            val engine = litertLmEngine
            val mpLlm = mediaPipeLlm

            if (engine != null) {
                val convConfig = ConversationConfig(
                    samplerConfig = SamplerConfig(
                        topK = 40,
                        topP = request.topP.toDouble(),
                        temperature = request.temperature.toDouble()
                    )
                )
                val conversation: Conversation = engine.createConversation(convConfig)
                val contentItems = mutableListOf<Content>()

                // WHY: Pass multimodal image bytes and audio bytes directly into LiteRT-LM's
                // Contents builder before the text prompt so Gemma 4 E2B processes all modalities together.
                request.messages.forEach { msg ->
                    msg.imageBytesList.forEach { imgBytes ->
                        contentItems.add(Content.ImageBytes(imgBytes))
                    }
                    msg.audioBytesList.forEach { audBytes ->
                        contentItems.add(Content.AudioBytes(audBytes))
                    }
                }
                contentItems.add(Content.Text(promptText))

                conversation.sendMessageAsync(
                    Contents.of(contentItems),
                    object : MessageCallback {
                        override fun onMessage(message: Message) {
                            val stepStart = System.nanoTime()
                            val chunk = message.toString()
                            if (chunk.isNotEmpty()) {
                                tokenCount++
                                val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
                                trySend(
                                    TokenStreamChunk(
                                        deltaText = chunk,
                                        isFinished = false,
                                        promptTokens = promptTokensEst,
                                        completionTokens = tokenCount,
                                        tokensPerSecond = tokenCount / elapsedSec
                                    )
                                )
                            }
                            optimizer.reportTokenStepDuration(System.nanoTime() - stepStart)
                        }

                        override fun onDone() {
                            val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
                            trySend(
                                TokenStreamChunk(
                                    deltaText = "",
                                    isFinished = true,
                                    promptTokens = promptTokensEst,
                                    completionTokens = tokenCount,
                                    tokensPerSecond = tokenCount / elapsedSec
                                )
                            )
                            try {
                                conversation.close()
                            } catch (_: Throwable) {
                            }
                            close()
                        }

                        override fun onError(throwable: Throwable) {
                            try {
                                conversation.close()
                            } catch (_: Throwable) {
                            }
                            close(throwable)
                        }
                    }
                )
            } else if (mpLlm != null) {
                val hasImages = request.messages.any { it.imageBytesList.isNotEmpty() }
                val hasAudio = request.messages.any { it.audioBytesList.isNotEmpty() }
                val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
                    .setTopK(40)
                    .setTopP(request.topP)
                    .setTemperature(request.temperature)
                    .setGraphOptions(
                        GraphOptions.builder()
                            .setEnableVisionModality(hasImages)
                            .setEnableAudioModality(hasAudio)
                            .build()
                    )
                    .build()
                val session = LlmInferenceSession.createFromOptions(mpLlm, sessionOptions)
                session.addQueryChunk(promptText)
                request.messages.forEach { msg ->
                    msg.imageBytesList.forEach { bytes ->
                        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.let { bmp ->
                            session.addImage(BitmapImageBuilder(bmp).build())
                        }
                    }
                    msg.audioBytesList.forEach { audioBytes ->
                        session.addAudio(audioBytes)
                    }
                }
                session.generateResponseAsync { partialResult, done ->
                    val stepStart = System.nanoTime()
                    if (partialResult.isNotEmpty()) {
                        tokenCount++
                        val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
                        trySend(
                            TokenStreamChunk(
                                deltaText = partialResult,
                                isFinished = false,
                                promptTokens = promptTokensEst,
                                completionTokens = tokenCount,
                                tokensPerSecond = tokenCount / elapsedSec
                            )
                        )
                    }
                    optimizer.reportTokenStepDuration(System.nanoTime() - stepStart)
                    if (done) {
                        val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
                        trySend(
                            TokenStreamChunk(
                                deltaText = "",
                                isFinished = true,
                                promptTokens = promptTokensEst,
                                completionTokens = tokenCount,
                                tokensPerSecond = tokenCount / elapsedSec
                            )
                        )
                        session.close()
                        close()
                    }
                }
            }
        } catch (e: Throwable) {
            close(e)
        }

        awaitClose { }
    }.flowOn(optimizer.inferenceDispatcher)

    suspend fun embedTexts(
        spec: AiModelSpec,
        modelFile: File,
        inputs: List<String>
    ): EmbeddingResponseData = withContext(optimizer.inferenceDispatcher) {
        val startMs = System.currentTimeMillis()
        if (activeEmbedderId != spec.id || textEmbedder == null) {
            unloadEmbedder()
            optimizer.mmapModelWeights(modelFile)
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(modelFile.absolutePath)
                .setDelegate(Delegate.CPU)
                .build()
            val options = TextEmbedder.TextEmbedderOptions.builder()
                .setBaseOptions(baseOptions)
                .setL2Normalize(true)
                .build()
            textEmbedder = TextEmbedder.createFromOptions(context, options)
            activeEmbedderId = spec.id
        }

        val embedder = textEmbedder ?: error("LiteRT TextEmbedder failed to initialize")
        val vectors = inputs.map { text ->
            val res = embedder.embed(text)
            val floatEmb = res.embeddingResult().embeddings().firstOrNull()?.floatEmbedding()
                ?: FloatArray(100)
            l2Normalize(floatEmb)
        }
        val tokenEst = inputs.sumOf { (it.length / 4).coerceAtLeast(1) }
        EmbeddingResponseData(
            model = spec.id,
            embeddings = vectors,
            promptTokens = tokenEst,
            latencyMs = (System.currentTimeMillis() - startMs).coerceAtLeast(1L)
        )
    }

    companion object {
        fun formatChatPrompt(messages: List<ChatMessageInput>): String {
            val sb = StringBuilder()
            for (msg in messages) {
                val roleTag = when (msg.role.lowercase()) {
                    "system" -> "user"
                    "assistant", "model" -> "model"
                    else -> "user"
                }
                sb.append("<start_of_turn>").append(roleTag).append("\n")
                sb.append(msg.text.trim()).append("<end_of_turn>\n")
            }
            sb.append("<start_of_turn>model\n")
            return sb.toString()
        }

        fun l2Normalize(vec: FloatArray): FloatArray {
            var sumSq = 0.0f
            for (v in vec) sumSq += v * v
            val norm = sqrt(sumSq.toDouble()).toFloat()
            if (norm <= 1e-12f) return vec
            return FloatArray(vec.size) { i -> vec[i] / norm }
        }
    }
}
