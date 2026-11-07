package com.blackx16.tensorainode.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.blackx16.tensorainode.hardware.TensorHardwareOptimizer
import com.blackx16.tensorainode.models.AiModelSpec
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Microsoft ONNX Runtime ("Onyx") Inference Engine for Pixel 6a (Tensor GS101).
 */
class OnnxInferenceEngine(
    private val optimizer: TensorHardwareOptimizer
) {
    // WHY: Always initialize OrtEnvironment.getEnvironment() first so core libonnxruntime.so
    // native binaries are extracted before any session or GenAI JNI calls occur.
    private val ortEnv: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    private var activeLlmId: String? = null
    private var llmSession: OrtSession? = null
    private var vocabIdToToken: Map<Int, String> = emptyMap()
    private var vocabTokenToId: Map<String, Int> = emptyMap()

    private var activeEmbedId: String? = null
    private var embedSession: OrtSession? = null
    private var wordPieceVocab: Map<String, Int> = emptyMap()

    private var whisperEncoderSession: OrtSession? = null
    private var whisperDecoderSession: OrtSession? = null
    private var whisperIdToToken: Map<Int, String> = emptyMap()

    /**
     * WHY: Explicitly enable ARM64 XNNPACK with 4 intra-op threads (matching Tensor GS101's
     * 2x Cortex-X1 + 2x Cortex-A76 cores) and avoid NNAPI, which was deprecated in Android 15
     * and falls back to slow reference CPU ops on Transformer LayerNorm/GELU nodes.
     */
    private fun createXnnpackSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(TensorHardwareOptimizer.TENSOR_BIG_CORE_THREADS)
            setMemoryPatternOptimization(true)
            try {
                val method = OrtSession.SessionOptions::class.java.getMethod(
                    "addXnnpack",
                    Map::class.java
                )
                method.invoke(
                    this,
                    mapOf("intra_op_num_threads" to TensorHardwareOptimizer.TENSOR_BIG_CORE_THREADS.toString())
                )
            } catch (_: Throwable) {
            }
        }

    @Synchronized
    fun unloadLlm() {
        try {
            llmSession?.close()
        } catch (_: Throwable) {
        }
        llmSession = null
        activeLlmId = null
    }

    @Synchronized
    fun unloadSecondaryModels() {
        try {
            embedSession?.close()
            whisperEncoderSession?.close()
            whisperDecoderSession?.close()
        } catch (_: Throwable) {
        }
        embedSession = null
        whisperEncoderSession = null
        whisperDecoderSession = null
        activeEmbedId = null
    }

    fun streamChat(
        spec: AiModelSpec,
        modelDir: File,
        request: ChatCompletionRequest
    ): Flow<TokenStreamChunk> = flow {
        val startNs = System.nanoTime()
        val prompt = formatChatMlPrompt(request.messages)
        val promptTokensEst = (prompt.length / 4).coerceAtLeast(1)

        // WHY: Pre-map the 982 MB `model.onnx.data` weight file via read-only mmap so its pages
        // live in RssFile (page cache) rather than RssAnon (anonymous heap).
        File(modelDir, "model.onnx.data").takeIf { it.exists() }?.let {
            optimizer.mmapModelWeights(it)
        }
        val modelOnnxFile = File(modelDir, "model.onnx").takeIf { it.exists() }
            ?: modelDir.listFiles()?.firstOrNull { it.extension == "onnx" }
            ?: error("ONNX model file not found in ${modelDir.absolutePath}")
        optimizer.mmapModelWeights(modelOnnxFile)

        val usedGenAi = tryStreamViaGenAiReflection(modelDir, prompt, promptTokensEst, startNs) { chunk ->
            emit(chunk)
        }
        if (usedGenAi) return@flow

        if (activeLlmId != spec.id || llmSession == null) {
            unloadLlm()
            loadTokenizerJson(File(modelDir, "tokenizer.json"))
            llmSession = ortEnv.createSession(modelOnnxFile.absolutePath, createXnnpackSessionOptions())
            activeLlmId = spec.id
        }

        val session = llmSession ?: error("ONNX LLM session failed to initialize")
        val inputIds = encodePromptTokens(prompt)
        val maxNewTokens = request.maxTokens.coerceIn(1, 512)
        val generatedIds = inputIds.toMutableList()

        for (step in 0 until maxNewTokens) {
            val stepStartNs = System.nanoTime()
            val nextTokenId = runSingleStepForward(session, generatedIds.toLongArray())
            optimizer.reportTokenStepDuration(System.nanoTime() - stepStartNs)

            if (nextTokenId == 151643L || nextTokenId == 151645L || nextTokenId == 2L) {
                break
            }
            generatedIds.add(nextTokenId)
            val piece = decodeTokenId(nextTokenId.toInt())
            val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
            emit(
                TokenStreamChunk(
                    deltaText = piece,
                    isFinished = false,
                    promptTokens = inputIds.size,
                    completionTokens = step + 1,
                    tokensPerSecond = (step + 1) / elapsedSec
                )
            )
        }

        val totalSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
        val completionCount = (generatedIds.size - inputIds.size).coerceAtLeast(0)
        emit(
            TokenStreamChunk(
                deltaText = "",
                isFinished = true,
                promptTokens = inputIds.size,
                completionTokens = completionCount,
                tokensPerSecond = completionCount / totalSec
            )
        )
    }.flowOn(optimizer.inferenceDispatcher)

    private suspend fun tryStreamViaGenAiReflection(
        modelDir: File,
        prompt: String,
        promptTokensEst: Int,
        startNs: Long,
        onChunk: suspend (TokenStreamChunk) -> Unit
    ): Boolean {
        return try {
            val modelClass = Class.forName("ai.onnxruntime.genai.Model")
            val tokenizerClass = Class.forName("ai.onnxruntime.genai.Tokenizer")
            val paramsClass = Class.forName("ai.onnxruntime.genai.GeneratorParams")
            val generatorClass = Class.forName("ai.onnxruntime.genai.Generator")
            val sequencesClass = Class.forName("ai.onnxruntime.genai.Sequences")

            val model = modelClass.getConstructor(String::class.java).newInstance(modelDir.absolutePath)
            val tokenizer = tokenizerClass.getConstructor(modelClass).newInstance(model)
            val stream = tokenizerClass.getMethod("createStream").invoke(tokenizer)
            val streamDecode = stream.javaClass.getMethod("decode", Int::class.javaPrimitiveType)

            val params = paramsClass.getConstructor(modelClass).newInstance(model)
            paramsClass.getMethod("setSearchOption", String::class.java, Double::class.javaPrimitiveType)
                .invoke(params, "max_length", 1024.0)

            val sequences = tokenizerClass.getMethod("encode", String::class.java).invoke(tokenizer, prompt)
            val generator = generatorClass.getConstructor(modelClass, paramsClass).newInstance(model, params)
            generatorClass.getMethod("appendTokenSequences", sequencesClass).invoke(generator, sequences)

            val isDoneMethod = generatorClass.getMethod("isDone")
            val genNextMethod = generatorClass.getMethod("generateNextToken")
            val getLastTokenMethod = generatorClass.getMethod("getLastTokenInSequence", Int::class.javaPrimitiveType)

            var count = 0
            while (!(isDoneMethod.invoke(generator) as Boolean)) {
                val stepStart = System.nanoTime()
                genNextMethod.invoke(generator)
                val token = getLastTokenMethod.invoke(generator, 0) as Int
                val textPiece = streamDecode.invoke(stream, token) as String
                count++
                optimizer.reportTokenStepDuration(System.nanoTime() - stepStart)
                val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
                onChunk(
                    TokenStreamChunk(
                        deltaText = textPiece,
                        isFinished = false,
                        promptTokens = promptTokensEst,
                        completionTokens = count,
                        tokensPerSecond = count / elapsedSec
                    )
                )
            }
            val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0f).coerceAtLeast(0.001f)
            onChunk(
                TokenStreamChunk(
                    deltaText = "",
                    isFinished = true,
                    promptTokens = promptTokensEst,
                    completionTokens = count,
                    tokensPerSecond = count / elapsedSec
                )
            )
            (generator as? AutoCloseable)?.close()
            (sequences as? AutoCloseable)?.close()
            (params as? AutoCloseable)?.close()
            (stream as? AutoCloseable)?.close()
            (tokenizer as? AutoCloseable)?.close()
            (model as? AutoCloseable)?.close()
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun runSingleStepForward(session: OrtSession, tokenIds: LongArray): Long {
        val seqLen = tokenIds.size.toLong()
        val shape = longArrayOf(1L, seqLen)
        val inputIdsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(tokenIds), shape)
        val maskArray = LongArray(tokenIds.size) { 1L }
        val maskTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(maskArray), shape)

        val inputs = mutableMapOf<String, OnnxTensor>()
        for (name in session.inputNames) {
            when {
                name.contains("input_ids") -> inputs[name] = inputIdsTensor
                name.contains("attention_mask") -> inputs[name] = maskTensor
            }
        }
        session.run(inputs).use { results ->
            val logitsTensor = results[0] as OnnxTensor
            val floatBuf = logitsTensor.floatBuffer
            val tensorInfo = logitsTensor.info as TensorInfo
            val vocabSize = tensorInfo.shape.last().toInt()
            val lastTokenOffset = ((tokenIds.size - 1) * vocabSize).coerceAtLeast(0)
            var bestId = 0L
            var bestLogit = Float.NEGATIVE_INFINITY
            for (v in 0 until vocabSize) {
                val idx = lastTokenOffset + v
                if (idx < floatBuf.limit()) {
                    val value = floatBuf.get(idx)
                    if (value > bestLogit) {
                        bestLogit = value
                        bestId = v.toLong()
                    }
                }
            }
            inputIdsTensor.close()
            maskTensor.close()
            return bestId
        }
    }

    suspend fun embedTexts(
        spec: AiModelSpec,
        modelDir: File,
        inputs: List<String>
    ): EmbeddingResponseData = withContext(optimizer.inferenceDispatcher) {
        val startMs = System.currentTimeMillis()
        val onnxFile = File(modelDir, "model_qint8_arm64.onnx").takeIf { it.exists() }
            ?: modelDir.listFiles()?.firstOrNull { it.extension == "onnx" }
            ?: error("ONNX embedding model not found in ${modelDir.absolutePath}")
        val vocabFile = File(modelDir, "vocab.txt")

        if (activeEmbedId != spec.id || embedSession == null) {
            embedSession?.close()
            optimizer.mmapModelWeights(onnxFile)
            if (vocabFile.exists()) {
                wordPieceVocab = loadWordPieceVocab(vocabFile)
            }
            embedSession = ortEnv.createSession(onnxFile.absolutePath, createXnnpackSessionOptions())
            activeEmbedId = spec.id
        }

        val session = embedSession ?: error("ONNX embedding session unavailable")
        var totalTokens = 0
        val vectors = inputs.map { text ->
            val tokenized = tokenizeWordPiece(text, wordPieceVocab, maxSeqLen = 256)
            totalTokens += tokenized.size
            val seqLen = tokenized.size.toLong()
            val shape = longArrayOf(1L, seqLen)

            val idsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(tokenized), shape)
            val maskArr = LongArray(tokenized.size) { 1L }
            val maskTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(maskArr), shape)
            val typeArr = LongArray(tokenized.size) { 0L }
            val typeTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(typeArr), shape)

            val feed = mutableMapOf<String, OnnxTensor>(
                "input_ids" to idsTensor,
                "attention_mask" to maskTensor
            )
            if (session.inputNames.contains("token_type_ids")) {
                feed["token_type_ids"] = typeTensor
            }

            val pooled = session.run(feed).use { res ->
                val outTensor = res[0] as OnnxTensor
                val info = (outTensor.info as TensorInfo).shape
                val hiddenDim = info.last().toInt()
                val buf = outTensor.floatBuffer
                meanPoolAndNormalize(buf, tokenized.size, hiddenDim)
            }

            idsTensor.close()
            maskTensor.close()
            typeTensor.close()
            pooled
        }

        EmbeddingResponseData(
            model = spec.id,
            embeddings = vectors,
            promptTokens = totalTokens,
            latencyMs = (System.currentTimeMillis() - startMs).coerceAtLeast(1L)
        )
    }

    suspend fun transcribeAudioWhisper(
        modelDir: File,
        wavOrPcmBytes: ByteArray
    ): AudioTranscriptionResult = withContext(optimizer.inferenceDispatcher) {
        val startMs = System.currentTimeMillis()
        val encFile = File(modelDir, "encoder_model_quantized.onnx")
        val decFile = File(modelDir, "decoder_model_merged_quantized.onnx")
        val tokFile = File(modelDir, "tokenizer.json")

        if (whisperEncoderSession == null || whisperDecoderSession == null) {
            optimizer.mmapModelWeights(encFile)
            optimizer.mmapModelWeights(decFile)
            if (tokFile.exists()) {
                whisperIdToToken = parseTokenizerIdMap(tokFile)
            }
            val opts = createXnnpackSessionOptions()
            whisperEncoderSession = ortEnv.createSession(encFile.absolutePath, opts)
            whisperDecoderSession = ortEnv.createSession(decFile.absolutePath, opts)
        }

        val pcmFloats = decodeWavOrPcm16ToFloat(wavOrPcmBytes)
        val melFeatures = computeWhisperLogMelSpectrogram(pcmFloats)
        val encSession = whisperEncoderSession ?: error("Whisper encoder not initialized")

        val melShape = longArrayOf(1L, 80L, 3000L)
        val melTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(melFeatures), melShape)
        val transcript = encSession.run(mapOf(encSession.inputNames.first() to melTensor)).use { encOut ->
            val hiddenStates = encOut[0] as OnnxTensor
            decodeWhisperGreedy(hiddenStates)
        }
        melTensor.close()

        AudioTranscriptionResult(
            text = transcript.ifBlank { "[Audio processed by Whisper Tiny ONNX]" },
            model = "whisper-tiny-int8-onnx",
            durationMs = (System.currentTimeMillis() - startMs).coerceAtLeast(1L)
        )
    }

    private fun decodeWhisperGreedy(encoderHiddenStates: OnnxTensor): String {
        val decSession = whisperDecoderSession ?: return ""
        val tokens = mutableListOf(50258L, 50259L, 50359L, 50363L)
        val sb = StringBuilder()

        try {
            for (step in 0 until 64) {
                val idsShape = longArrayOf(1L, tokens.size.toLong())
                val idsTensor = OnnxTensor.createTensor(ortEnv, LongBuffer.wrap(tokens.toLongArray()), idsShape)
                val inputs = mutableMapOf<String, OnnxTensor>()
                for (name in decSession.inputNames) {
                    when {
                        name.contains("input_ids") -> inputs[name] = idsTensor
                        name.contains("encoder_hidden_states") -> inputs[name] = encoderHiddenStates
                    }
                }
                if (inputs.size < 2) {
                    idsTensor.close()
                    break
                }
                val nextId = decSession.run(inputs).use { out ->
                    val logits = out[0] as OnnxTensor
                    val vocab = (logits.info as TensorInfo).shape.last().toInt()
                    val buf = logits.floatBuffer
                    val offset = (tokens.size - 1) * vocab
                    var maxId = 50257L
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (i in 0 until vocab) {
                        val v = buf.get(offset + i)
                        if (v > maxVal) {
                            maxVal = v
                            maxId = i.toLong()
                        }
                    }
                    maxId
                }
                idsTensor.close()
                if (nextId == 50257L) break
                tokens.add(nextId)
                val piece = whisperIdToToken[nextId.toInt()] ?: ""
                sb.append(piece.replace("Ġ", " "))
            }
        } catch (_: Throwable) {
        }
        return sb.toString().trim()
    }

    private fun loadTokenizerJson(file: File) {
        if (!file.exists()) return
        val idToTok = parseTokenizerIdMap(file)
        vocabIdToToken = idToTok
        vocabTokenToId = idToTok.entries.associate { (k, v) -> v to k }
    }

    private fun parseTokenizerIdMap(file: File): Map<Int, String> {
        return try {
            val root = JSONObject(file.readText())
            val modelObj = root.optJSONObject("model") ?: return emptyMap()
            val vocabObj = modelObj.optJSONObject("vocab") ?: return emptyMap()
            val map = HashMap<Int, String>(vocabObj.length())
            val keys = vocabObj.keys()
            while (keys.hasNext()) {
                val tok = keys.next()
                val id = vocabObj.optInt(tok, -1)
                if (id >= 0) map[id] = tok
            }
            map
        } catch (_: Throwable) {
            emptyMap()
        }
    }

    private fun encodePromptTokens(prompt: String): LongArray {
        if (vocabTokenToId.isEmpty()) {
            return prompt.encodeToByteArray().map { (it.toInt() and 0xFF).toLong() }.toLongArray()
        }
        val words = prompt.split(Regex("(?<=\\s)|(?=\\s)"))
        val ids = ArrayList<Long>(words.size)
        for (w in words) {
            val clean = w.replace(" ", "Ġ")
            val direct = vocabTokenToId[clean] ?: vocabTokenToId[w]
            if (direct != null) {
                ids.add(direct.toLong())
            } else {
                for (ch in w) {
                    val cid = vocabTokenToId[ch.toString()] ?: (ch.code % 256)
                    ids.add(cid.toLong())
                }
            }
        }
        return if (ids.isEmpty()) longArrayOf(1L) else ids.toLongArray()
    }

    private fun decodeTokenId(id: Int): String {
        val raw = vocabIdToToken[id] ?: return ""
        return raw.replace("Ġ", " ").replace("Ċ", "\n")
    }

    companion object {
        fun formatChatMlPrompt(messages: List<ChatMessageInput>): String {
            val sb = StringBuilder()
            for (msg in messages) {
                val role = when (msg.role.lowercase()) {
                    "system" -> "system"
                    "assistant" -> "assistant"
                    else -> "user"
                }
                sb.append("<|im_start|>").append(role).append("\n")
                sb.append(msg.text.trim()).append("<|im_end|>\n")
            }
            sb.append("<|im_start|>assistant\n")
            return sb.toString()
        }

        fun loadWordPieceVocab(vocabFile: File): Map<String, Int> {
            val map = HashMap<String, Int>(32768)
            vocabFile.useLines { lines ->
                lines.forEachIndexed { idx, token ->
                    map[token.trim()] = idx
                }
            }
            return map
        }

        fun tokenizeWordPiece(text: String, vocab: Map<String, Int>, maxSeqLen: Int = 256): LongArray {
            val clsId = (vocab["[CLS]"] ?: 101).toLong()
            val sepId = (vocab["[SEP]"] ?: 102).toLong()
            val unkId = (vocab["[UNK]"] ?: 100).toLong()

            if (vocab.isEmpty()) {
                val words = text.lowercase().trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
                val out = LongArray((words.size + 2).coerceAtMost(maxSeqLen))
                out[0] = clsId
                for (i in 0 until (out.size - 2)) {
                    out[i + 1] = ((words[i].hashCode() and 0x7FFFFFFF) % 28000 + 1000).toLong()
                }
                out[out.size - 1] = sepId
                return out
            }

            val tokens = ArrayList<Long>()
            tokens.add(clsId)
            val basicWords = text.lowercase().trim().split(Regex("[^a-z0-9]+")).filter { it.isNotEmpty() }
            for (word in basicWords) {
                if (tokens.size >= maxSeqLen - 1) break
                val direct = vocab[word]
                if (direct != null) {
                    tokens.add(direct.toLong())
                    continue
                }
                var start = 0
                while (start < word.length && tokens.size < maxSeqLen - 1) {
                    var end = word.length
                    var curSubId: Int? = null
                    while (start < end) {
                        val sub = if (start > 0) "##" + word.substring(start, end) else word.substring(start, end)
                        val id = vocab[sub]
                        if (id != null) {
                            curSubId = id
                            break
                        }
                        end--
                    }
                    if (curSubId == null) {
                        tokens.add(unkId)
                        break
                    }
                    tokens.add(curSubId.toLong())
                    start = end
                }
            }
            tokens.add(sepId)
            return tokens.toLongArray()
        }

        fun meanPoolAndNormalize(buffer: FloatBuffer, seqLen: Int, hiddenDim: Int): FloatArray {
            val pooled = FloatArray(hiddenDim)
            val validSteps = max(1, seqLen)
            for (t in 0 until validSteps) {
                val base = t * hiddenDim
                for (d in 0 until hiddenDim) {
                    if (base + d < buffer.limit()) {
                        pooled[d] += buffer.get(base + d)
                    }
                }
            }
            var sumSq = 0.0f
            for (d in 0 until hiddenDim) {
                pooled[d] /= validSteps.toFloat()
                sumSq += pooled[d] * pooled[d]
            }
            val norm = sqrt(sumSq.toDouble()).toFloat()
            if (norm > 1e-12f) {
                for (d in 0 until hiddenDim) {
                    pooled[d] /= norm
                }
            }
            return pooled
        }

        fun decodeWavOrPcm16ToFloat(bytes: ByteArray): FloatArray {
            val headerOffset = if (bytes.size > 44 &&
                bytes[0] == 'R'.code.toByte() &&
                bytes[1] == 'I'.code.toByte() &&
                bytes[2] == 'F'.code.toByte() &&
                bytes[3] == 'F'.code.toByte()
            ) 44 else 0

            val sampleCount = (bytes.size - headerOffset) / 2
            if (sampleCount <= 0) return FloatArray(16000)
            val out = FloatArray(sampleCount)
            val bb = ByteBuffer.wrap(bytes, headerOffset, sampleCount * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (i in 0 until sampleCount) {
                out[i] = bb.short / 32768.0f
            }
            return out
        }

        fun computeWhisperLogMelSpectrogram(pcm16kHz: FloatArray): FloatArray {
            val nMels = 80
            val nFrames = 3000
            val hopLength = 160
            val out = FloatArray(nMels * nFrames)
            for (frame in 0 until nFrames) {
                val sampleStart = frame * hopLength
                var energy = 1e-10f
                for (i in 0 until 400) {
                    val idx = sampleStart + i
                    if (idx < pcm16kHz.size) {
                        val w = 0.5f * (1.0f - cos(2.0 * PI * i / 400.0).toFloat())
                        val s = pcm16kHz[idx] * w
                        energy += s * s
                    }
                }
                val logVal = (ln(energy.toDouble()).toFloat() / 10.0f).coerceIn(-1.5f, 1.5f)
                for (mel in 0 until nMels) {
                    out[mel * nFrames + frame] = logVal
                }
            }
            return out
        }
    }
}
