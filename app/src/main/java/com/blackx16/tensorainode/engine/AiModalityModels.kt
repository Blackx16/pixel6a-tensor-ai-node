package com.blackx16.tensorainode.engine

data class ChatMessageInput(
    val role: String,
    val text: String,
    val imageBytesList: List<ByteArray> = emptyList(),
    val audioBytesList: List<ByteArray> = emptyList()
)

data class ChatCompletionRequest(
    val model: String,
    val messages: List<ChatMessageInput>,
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val maxTokens: Int = 512,
    val stream: Boolean = false
)

data class TokenStreamChunk(
    val deltaText: String,
    val isFinished: Boolean = false,
    val promptTokens: Int = 0,
    val completionTokens: Int = 0,
    val tokensPerSecond: Float = 0f
)

data class EmbeddingResponseData(
    val model: String,
    val embeddings: List<FloatArray>,
    val promptTokens: Int,
    val latencyMs: Long
)

data class AudioTranscriptionResult(
    val text: String,
    val model: String,
    val durationMs: Long
)
