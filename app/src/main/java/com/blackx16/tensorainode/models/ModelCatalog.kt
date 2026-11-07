package com.blackx16.tensorainode.models

enum class RuntimeBackend(val displayName: String) {
    LITERT("Google LiteRT / LiteRT-LM"),
    ONNX("ONNX Runtime / XNNPACK"),
    SYSTEM_TENSOR("Pixel Tensor Hardware")
}

enum class ModelCapability {
    TEXT,
    IMAGE_VISION,
    VOICE_ASR,
    VOICE_TTS,
    EMBEDDINGS
}

data class DownloadableFileSpec(
    val relativePath: String,
    val downloadUrl: String,
    val expectedSizeBytes: Long
)

data class AiModelSpec(
    val id: String,
    val title: String,
    val runtime: RuntimeBackend,
    val capabilities: Set<ModelCapability>,
    val quantization: String,
    val estimatedRamMb: Int,
    val description: String,
    val requiresHfToken: Boolean = false,
    val files: List<DownloadableFileSpec>
) {
    val totalDownloadBytes: Long
        get() = files.sumOf { it.expectedSizeBytes }

    val totalDownloadMb: Int
        get() = (totalDownloadBytes / (1024L * 1024L)).toInt()
}

/**
 * Curated post-install downloadable model catalog tailored for the Google Pixel 6a (6 GB RAM).
 *
 * WHY: Zero models are bundled inside the APK so installation is instant and lightweight.
 * All models are fetched over HTTPS after installation and stored in `filesDir/models/`.
 */
object ModelCatalog {

    const val PRIMARY_LITERT_LLM_ID = "gemma-4-e2b-it-litert"
    const val PRIMARY_ONNX_LLM_ID = "qwen2.5-1.5b-instruct-onnx"
    const val LITERT_FAST_LLM_ID = "gemma-3-1b-it-litert"
    const val LITERT_EMBED_ID = "universal-sentence-encoder-litert"
    const val ONNX_EMBED_ID = "all-minilm-l6-v2-onnx"
    const val ONNX_WHISPER_ID = "whisper-tiny-int8-onnx"

    val BUILTIN_MODELS: List<AiModelSpec> = listOf(
        // WHY: Chosen as Primary LLM #1 (LiteRT) because Gemma 4 E2B IT uses mixed 2/4/8-bit
        // quantization with mmap'd embeddings, requiring only ~820 MB of active GPU RAM on the
        // Mali-G78 MP20 while natively supporting Text, Image Vision, and Audio inputs.
        AiModelSpec(
            id = PRIMARY_LITERT_LLM_ID,
            title = "Gemma 4 E2B IT (LiteRT-LM GPU Multimodal)",
            runtime = RuntimeBackend.LITERT,
            capabilities = setOf(
                ModelCapability.TEXT,
                ModelCapability.IMAGE_VISION,
                ModelCapability.VOICE_ASR
            ),
            quantization = "Mixed 2/4/8-bit + mmap Embeddings",
            estimatedRamMb = 820,
            description = "Primary LiteRT LLM for Pixel 6a. Supports Text, Image Vision, and Audio input on Mali-G78 MP20 GPU with only ~820 MB active GPU RAM.",
            requiresHfToken = false,
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "gemma-4-e2b-it/gemma-4-E2B-it-gpu.litertlm",
                    downloadUrl = "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm",
                    expectedSizeBytes = 2_010_000_000L
                )
            )
        ),

        // WHY: Chosen as Primary LLM #2 (ONNX Runtime) because Qwen2.5-1.5B-Instruct INT4 fits in
        // ~1.35 GB RAM (avoiding the OOM kills seen with 3.8B models on 6 GB devices) and executes
        // rapidly on the Tensor GS101's 4 big cores via ARM64 XNNPACK.
        AiModelSpec(
            id = PRIMARY_ONNX_LLM_ID,
            title = "Qwen2.5-1.5B-Instruct (ONNX INT4 Mobile)",
            runtime = RuntimeBackend.ONNX,
            capabilities = setOf(ModelCapability.TEXT),
            quantization = "INT4 RTN + ARM64 XNNPACK",
            estimatedRamMb = 1350,
            description = "Primary ONNX Runtime LLM for Pixel 6a. Runs on 4 Tensor GS101 big cores (2x Cortex-X1 + 2x Cortex-A76) via ARM64 XNNPACK.",
            requiresHfToken = false,
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "qwen2.5-1.5b-onnx/model.onnx",
                    downloadUrl = "https://huggingface.co/pdnhan/Qwen2.5-1.5B-Instruct-mobile/resolve/main/onnx/model.onnx",
                    expectedSizeBytes = 224_000L
                ),
                DownloadableFileSpec(
                    relativePath = "qwen2.5-1.5b-onnx/model.onnx.data",
                    downloadUrl = "https://huggingface.co/pdnhan/Qwen2.5-1.5B-Instruct-mobile/resolve/main/onnx/model.onnx.data",
                    expectedSizeBytes = 982_000_000L
                ),
                DownloadableFileSpec(
                    relativePath = "qwen2.5-1.5b-onnx/genai_config.json",
                    downloadUrl = "https://huggingface.co/pdnhan/Qwen2.5-1.5B-Instruct-mobile/resolve/main/onnx/genai_config.json",
                    expectedSizeBytes = 1_580L
                ),
                DownloadableFileSpec(
                    relativePath = "qwen2.5-1.5b-onnx/tokenizer.json",
                    downloadUrl = "https://huggingface.co/pdnhan/Qwen2.5-1.5B-Instruct-mobile/resolve/main/onnx/tokenizer.json",
                    expectedSizeBytes = 11_400_000L
                ),
                DownloadableFileSpec(
                    relativePath = "qwen2.5-1.5b-onnx/tokenizer_config.json",
                    downloadUrl = "https://huggingface.co/pdnhan/Qwen2.5-1.5B-Instruct-mobile/resolve/main/onnx/tokenizer_config.json",
                    expectedSizeBytes = 692L
                )
            )
        ),

        // WHY: Purpose-built ARM64 QINT8 ONNX build of all-MiniLM-L6-v2 (only 23 MB disk, ~48 MB RAM)
        // so embeddings can remain co-resident in memory alongside an active LLM on 6 GB RAM.
        AiModelSpec(
            id = ONNX_EMBED_ID,
            title = "all-MiniLM-L6-v2 (ONNX ARM64 INT8 Embeddings)",
            runtime = RuntimeBackend.ONNX,
            capabilities = setOf(ModelCapability.EMBEDDINGS),
            quantization = "Dynamic QINT8 ARM64",
            estimatedRamMb = 48,
            description = "384-dimensional semantic text embeddings in <15 ms on Tensor GS101 XNNPACK.",
            requiresHfToken = false,
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "minilm-l6-v2-onnx/model_qint8_arm64.onnx",
                    downloadUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_qint8_arm64.onnx",
                    expectedSizeBytes = 23_000_000L
                ),
                DownloadableFileSpec(
                    relativePath = "minilm-l6-v2-onnx/vocab.txt",
                    downloadUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
                    expectedSizeBytes = 232_000L
                )
            )
        ),

        // WHY: Provides a zero-auth Google Cloud Storage LiteRT .tflite embedding option with built-in tokenizer metadata.
        AiModelSpec(
            id = LITERT_EMBED_ID,
            title = "Universal Sentence Encoder (LiteRT Embeddings)",
            runtime = RuntimeBackend.LITERT,
            capabilities = setOf(ModelCapability.EMBEDDINGS),
            quantization = "FP32 Compact .tflite",
            estimatedRamMb = 35,
            description = "100-dimensional Google LiteRT / MediaPipe TextEmbedder model with built-in tokenizer metadata.",
            requiresHfToken = false,
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "use-litert/universal_sentence_encoder.tflite",
                    downloadUrl = "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/latest/universal_sentence_encoder.tflite",
                    expectedSizeBytes = 25_200_000L
                )
            )
        ),

        // WHY: INT8 quantized Whisper Tiny (40.8 MB total) enables deterministic offline speech transcription
        // for the OpenAI `/v1/audio/transcriptions` endpoint without requiring cloud speech APIs.
        AiModelSpec(
            id = ONNX_WHISPER_ID,
            title = "Whisper Tiny (ONNX INT8 Speech-to-Text)",
            runtime = RuntimeBackend.ONNX,
            capabilities = setOf(ModelCapability.VOICE_ASR),
            quantization = "INT8 Quantized Encoder + Decoder",
            estimatedRamMb = 130,
            description = "Offline ONNX speech-to-text transcription for /v1/audio/transcriptions and voice chat.",
            requiresHfToken = false,
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "whisper-tiny-onnx/encoder_model_quantized.onnx",
                    downloadUrl = "https://huggingface.co/Xenova/whisper-tiny/resolve/main/onnx/encoder_model_quantized.onnx",
                    expectedSizeBytes = 10_100_000L
                ),
                DownloadableFileSpec(
                    relativePath = "whisper-tiny-onnx/decoder_model_merged_quantized.onnx",
                    downloadUrl = "https://huggingface.co/Xenova/whisper-tiny/resolve/main/onnx/decoder_model_merged_quantized.onnx",
                    expectedSizeBytes = 30_700_000L
                ),
                DownloadableFileSpec(
                    relativePath = "whisper-tiny-onnx/tokenizer.json",
                    downloadUrl = "https://huggingface.co/Xenova/whisper-tiny/resolve/main/tokenizer.json",
                    expectedSizeBytes = 2_400_000L
                )
            )
        ),

        AiModelSpec(
            id = LITERT_FAST_LLM_ID,
            title = "Gemma 3 1B IT (LiteRT-LM INT4 Fast Text)",
            runtime = RuntimeBackend.LITERT,
            capabilities = setOf(ModelCapability.TEXT),
            quantization = "INT4 (4K KV Cache)",
            estimatedRamMb = 850,
            description = "Compact 584 MB Gemma 3 1B model for ultra-fast text generation (requires Hugging Face token with Gemma license accepted).",
            requiresHfToken = true,
            files = listOf(
                DownloadableFileSpec(
                    relativePath = "gemma-3-1b-it/gemma3-1b-it-int4.litertlm",
                    downloadUrl = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.litertlm",
                    expectedSizeBytes = 584_000_000L
                )
            )
        )
    )

    fun findById(id: String): AiModelSpec? =
        BUILTIN_MODELS.firstOrNull { it.id.equals(id, ignoreCase = true) }
}
