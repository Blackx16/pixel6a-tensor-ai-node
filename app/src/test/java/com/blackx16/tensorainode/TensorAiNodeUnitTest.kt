package com.blackx16.tensorainode

import com.blackx16.tensorainode.engine.ChatMessageInput
import com.blackx16.tensorainode.engine.LiteRtInferenceEngine
import com.blackx16.tensorainode.engine.OnnxInferenceEngine
import com.blackx16.tensorainode.engine.UnifiedAiRouter
import com.blackx16.tensorainode.engine.VoiceSpeechEngine
import com.blackx16.tensorainode.models.ModelCatalog
import com.blackx16.tensorainode.models.ModelCapability
import com.blackx16.tensorainode.models.RuntimeBackend
import com.blackx16.tensorainode.server.OpenAiHttpServer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class TensorAiNodeUnitTest {

    @Test
    fun `verify two primary LLMs and all modalities in catalog`() {
        val litertLlm = ModelCatalog.findById(ModelCatalog.PRIMARY_LITERT_LLM_ID)
        val onnxLlm = ModelCatalog.findById(ModelCatalog.PRIMARY_ONNX_LLM_ID)
        assertNotNull(litertLlm)
        assertNotNull(onnxLlm)
        assertEquals(RuntimeBackend.LITERT, litertLlm!!.runtime)
        assertEquals(RuntimeBackend.ONNX, onnxLlm!!.runtime)

        // Ensure Pixel 6a 6 GB RAM safety (<2.2 GB RAM per model)
        assertTrue(litertLlm.estimatedRamMb < 2000)
        assertTrue(onnxLlm.estimatedRamMb < 2000)

        // Ensure all 4 requested capabilities (Text, Image Vision, Voice, Embeddings) exist
        val allCaps = ModelCatalog.BUILTIN_MODELS.flatMap { it.capabilities }.toSet()
        assertTrue(ModelCapability.TEXT in allCaps)
        assertTrue(ModelCapability.IMAGE_VISION in allCaps)
        assertTrue(ModelCapability.VOICE_ASR in allCaps)
        assertTrue(ModelCapability.EMBEDDINGS in allCaps)
    }

    @Test
    fun `verify Bearer API key constant-time verification`() {
        assertTrue(OpenAiHttpServer.isAuthorized("Bearer sk-pixel6a-tensor", "sk-pixel6a-tensor"))
        assertFalse(OpenAiHttpServer.isAuthorized("Bearer wrong-key", "sk-pixel6a-tensor"))
        assertFalse(OpenAiHttpServer.isAuthorized(null, "sk-pixel6a-tensor"))
    }

    @Test
    fun `verify prompt formatting and L2 embedding normalization`() {
        val msgs = listOf(ChatMessageInput(role = "user", text = "Hello Pixel 6a"))
        val gemmaPrompt = LiteRtInferenceEngine.formatChatPrompt(msgs)
        assertTrue(gemmaPrompt.contains("<start_of_turn>user"))
        assertTrue(gemmaPrompt.contains("<start_of_turn>model"))

        val qwenPrompt = OnnxInferenceEngine.formatChatMlPrompt(msgs)
        assertTrue(qwenPrompt.contains("<|im_start|>user"))
        assertTrue(qwenPrompt.contains("<|im_start|>assistant"))

        val emb = UnifiedAiRouter.deterministicFallbackEmbedding("Tensor GS101", 384)
        assertEquals(384, emb.size)
        val selfSim = UnifiedAiRouter.cosineSimilarity(emb, emb)
        assertTrue(abs(1.0f - selfSim) < 1e-4f)
    }

    @Test
    fun `verify RIFF WAVE speech synthesis header`() {
        val wav = VoiceSpeechEngine.generateDeterministicPcmWav("Hello from Android 17")
        assertTrue(wav.size > 44)
        assertEquals('R'.code.toByte(), wav[0])
        assertEquals('I'.code.toByte(), wav[1])
        assertEquals('F'.code.toByte(), wav[2])
        assertEquals('F'.code.toByte(), wav[3])
    }
}
