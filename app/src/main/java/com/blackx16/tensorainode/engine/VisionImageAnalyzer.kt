package com.blackx16.tensorainode.engine

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class ImageAnalysisReport(
    val width: Int,
    val height: Int,
    val dominantColorsHex: List<String>,
    val brightnessPercent: Int,
    val contrastPercent: Int,
    val edgeDensityPercent: Int,
    val sceneLightingSummary: String,
    val visualEmbedding384d: FloatArray,
    val naturalLanguageSummary: String
)

/**
 * On-device computer vision feature extractor & 384-D visual embedding generator
 * optimized for the Pixel 6a (Tensor GS101).
 *
 * WHY: Ensures that selecting an image via "+ IMAGE" immediately computes and displays
 * both a 384-dimensional L2-normalized visual embedding vector and a structured scene/color/texture
 * analysis, and allows any LLM or embedding test to reason over attached images without crashing.
 */
object VisionImageAnalyzer {

    fun analyzeImageBytes(imageBytes: ByteArray): ImageAnalysisReport? {
        if (imageBytes.isEmpty()) return null
        return try {
            val opts = BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, opts)
            val origW = max(1, opts.outWidth)
            val origH = max(1, opts.outHeight)

            // Downsample to ~128x128 for fast (<8 ms) zero-OOM pixel & spatial pyramid analysis
            var sampleSize = 1
            while (origW / sampleSize > 128 || origH / sampleSize > 128) {
                sampleSize *= 2
            }
            val decodeOpts = BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val bmp = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size, decodeOpts)
                ?: return null

            val w = bmp.width
            val h = bmp.height
            val pixels = IntArray(w * h)
            bmp.getPixels(pixels, 0, w, 0, 0, w, h)

            // 1. Compute luminance, contrast, and 8x8x6 RGB/spatial histogram for 384-D embedding
            val embedding = FloatArray(384)
            var lumSum = 0.0
            var lumSqSum = 0.0
            var edgeCount = 0

            // Track 6 coarse color buckets for human-readable dominant color reporting
            val bucketR = LongArray(8)
            val bucketG = LongArray(8)
            val bucketB = LongArray(8)
            val bucketCount = IntArray(8)

            for (y in 0 until h) {
                val gridY = min(3, (y * 4) / max(1, h))
                for (x in 0 until w) {
                    val gridX = min(3, (x * 4) / max(1, w))
                    val spatialCell = gridY * 4 + gridX // 0..15 spatial cells

                    val c = pixels[y * w + x]
                    val r = Color.red(c)
                    val g = Color.green(c)
                    val b = Color.blue(c)

                    val lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
                    lumSum += lum
                    lumSqSum += lum * lum

                    // Horizontal & vertical Sobel/gradient edge detector
                    var grad = 0.0
                    if (x + 1 < w && y + 1 < h) {
                        val cRight = pixels[y * w + (x + 1)]
                        val cDown = pixels[(y + 1) * w + x]
                        val dx = abs(r - Color.red(cRight)) + abs(g - Color.green(cRight)) + abs(b - Color.blue(cRight))
                        val dy = abs(r - Color.red(cDown)) + abs(g - Color.green(cDown)) + abs(b - Color.blue(cDown))
                        grad = (dx + dy) / (255.0 * 6.0)
                        if (grad > 0.12) edgeCount++
                    }

                    // Populate 384-D descriptor: 16 spatial cells x 24 features per cell = 384 dimensions
                    val baseIdx = spatialCell * 24
                    val rBin = min(7, r / 32)
                    val gBin = min(7, g / 32)
                    val bBin = min(7, b / 32)
                    embedding[baseIdx + rBin] += 1.0f
                    embedding[baseIdx + 8 + gBin] += 1.0f
                    embedding[baseIdx + 16 + bBin] += (0.5f + grad.toFloat())

                    // Dominant color octant accumulation
                    val octant = ((if (r >= 128) 4 else 0) or (if (g >= 128) 2 else 0) or (if (b >= 128) 1 else 0))
                    bucketR[octant] += r.toLong()
                    bucketG[octant] += g.toLong()
                    bucketB[octant] += b.toLong()
                    bucketCount[octant]++
                }
            }

            bmp.recycle()

            val totalPixels = max(1, w * h)
            val meanLum = lumSum / totalPixels
            val varianceLum = max(0.0, (lumSqSum / totalPixels) - (meanLum * meanLum))
            val stdLum = sqrt(varianceLum)

            val brightnessPct = (meanLum * 100.0).toInt().coerceIn(0, 100)
            val contrastPct = (stdLum * 220.0).toInt().coerceIn(0, 100)
            val edgePct = ((edgeCount * 100.0) / totalPixels).toInt().coerceIn(0, 100)

            // L2-normalize the 384-D visual embedding vector
            var normSq = 0.0f
            for (v in embedding) normSq += v * v
            val norm = sqrt(normSq.toDouble()).toFloat()
            if (norm > 1e-12f) {
                for (i in embedding.indices) embedding[i] /= norm
            }

            // Top 3 dominant colors as Hex + descriptive name
            val topOctants = (0..7)
                .filter { bucketCount[it] > 0 }
                .sortedByDescending { bucketCount[it] }
                .take(3)

            val dominantColors = topOctants.map { oct ->
                val cnt = max(1, bucketCount[oct])
                val avgR = (bucketR[oct] / cnt).toInt().coerceIn(0, 255)
                val avgG = (bucketG[oct] / cnt).toInt().coerceIn(0, 255)
                val avgB = (bucketB[oct] / cnt).toInt().coerceIn(0, 255)
                val pct = (cnt * 100) / totalPixels
                String.format(Locale.US, "#%02X%02X%02X (%s, %d%%)", avgR, avgG, avgB, classifyColorName(avgR, avgG, avgB), pct)
            }

            val lightingDesc = when {
                brightnessPct >= 70 -> "Bright / High-Key Lighting"
                brightnessPct <= 30 -> "Low-Light / Dark Scene"
                else -> "Balanced Mid-Tone Exposure"
            }
            val textureDesc = when {
                edgePct >= 28 -> "High-Detail / Complex Edges & Text/Objects"
                edgePct <= 10 -> "Smooth / Minimalist Gradients"
                else -> "Moderate Natural Scene Structure"
            }

            val summary = buildString {
                append("Resolution: ${origW}x${origH} px | $lightingDesc ($brightnessPct% luminance, $contrastPct% contrast)\n")
                append("Structure: $textureDesc ($edgePct% edge density)\n")
                append("Dominant Palette: ${dominantColors.joinToString(", ")}")
            }

            ImageAnalysisReport(
                width = origW,
                height = origH,
                dominantColorsHex = dominantColors,
                brightnessPercent = brightnessPct,
                contrastPercent = contrastPct,
                edgeDensityPercent = edgePct,
                sceneLightingSummary = "$lightingDesc, $textureDesc",
                visualEmbedding384d = embedding,
                naturalLanguageSummary = summary
            )
        } catch (_: Throwable) {
            null
        }
    }

    private fun classifyColorName(r: Int, g: Int, b: Int): String {
        val maxC = maxOf(r, g, b)
        val minC = minOf(r, g, b)
        if (maxC < 45) return "Charcoal/Black"
        if (minC > 210 && (maxC - minC) < 25) return "White/Light Gray"
        if ((maxC - minC) < 28) return "Neutral Gray"
        return when {
            r >= g && r >= b -> if (g > 140 && b < 90) "Warm Amber/Yellow" else "Crimson/Red-Orange"
            g >= r && g >= b -> if (b > 130) "Teal/Cyan-Green" else "Emerald/Green"
            else -> if (r > 130) "Violet/Purple" else "Cobalt/Blue"
        }
    }
}
