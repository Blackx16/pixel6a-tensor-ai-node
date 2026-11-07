package com.blackx16.tensorainode.engine

import android.content.Context
import com.blackx16.tensorainode.hardware.TensorHardwareOptimizer
import com.blackx16.tensorainode.models.AiModelSpec
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.text.textembedder.TextEmbedder
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.sqrt

/**
 * Google LiteRT ("Lightartie") & MediaPipe GenAI Inference Engine for Pixel 6a
 * (Google Tensor GS101 + Mali-G78 MP20 GPU).
 */
class LiteRtInferenceEngine(
    private val context: Context,
    private val optimizer: TensorHardwareOptimizer
) {
    private var activeModelId: String? = null
    private var mediaPipeLlm: LlmInference? = null
    private var textEmbedder: TextEmbedder? = null
    private var activeEmbedderId: String? = null

    @Synchronized
    fun unloadLlm() {
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
     * the LiteRT / MediaPipe LLM session, and caps `maxTokens = 2048` so the KV cache never
     * breaches the Pixel 6a's 6 GB RAM ceiling.
     */
    fun streamChat(
        spec: AiModelSpec,
        modelFile: File,
        request: ChatCompletionRequest
    ): Flow<TokenStreamChunk> = callbackFlow {
        val startNs = System.nanoTime()
        var tokenCount = 0

        try {
            optimizer.mmapModelWeights(modelFile)
            val promptText = formatChatPrompt(request.messages)
            val promptTokensEst = (promptText.length / 4).coerceAtLeast(1)

            // Recreate LlmInference with streaming result listener directed to this callbackFlow
            unloadLlm()
            val options = LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelFile.absolutePath)
                .setMaxTokens(request.maxTokens.coerceIn(64, 2048))
                .setTopK(40)
                .setTemperature(request.temperature)
                .setResultListener { partialResult, done ->
                    val stepStart = System.nanoTime()
                    if (!partialResult.isNullOrEmpty()) {
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
                        close()
                    }
                }
                .setErrorListener { err ->
                    close(err)
                }
                .build()

            val llm = LlmInference.createFromOptions(context, options)
            mediaPipeLlm = llm
            activeModelId = spec.id
            llm.generateResponseAsync(promptText)
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
                if (msg.imageBytesList.isNotEmpty()) {
                    sb.append("[Image attached: ${msg.imageBytesList.size} frame(s)]\n")
                }
                if (msg.audioBytesList.isNotEmpty()) {
                    sb.append("[Audio clip attached]\n")
                }
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
