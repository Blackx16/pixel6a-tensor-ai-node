# Pixel 6a TensorAI Node (`pixel6a-tensor-ai-node`)

An ultra-lightweight native **Android 17 (`API 37`, build `DP11.261016.003`)** local AI server and on-device multimodal runtime built specifically for the **[Google Pixel 6a](https://en.wikipedia.org/wiki/Pixel_6a)** (`bluejay`, **[Google Tensor GS101](https://en.wikipedia.org/wiki/Google_Tensor)**, **[Mali-G78 MP20 GPU](https://www.gsmarena.com/google_pixel_6a-11229.php)**, **6 GB LPDDR5/LPDDR4X RAM**, **128 GB UFS 3.1 storage**).

## Research Documentation

- [`docs/RESEARCH_ANDROID17_TENSOR_GS101.md`](docs/RESEARCH_ANDROID17_TENSOR_GS101.md) — Hardware & OS architecture analysis for Android 17 (`DP11.261016.003`, Kernel `6.1.172-android14-11`), `ACCESS_LOCAL_NETWORK`, `MemoryLimiter:AnonSwap`, 16 KB page alignment, `0xF0` Cortex-X1/A76 thread affinity pinning, `PerformanceHintManager`, and read-only `mmap` weight streaming.
- [`docs/RESEARCH_LITERT_AND_ONNX_MODELS.md`](docs/RESEARCH_LITERT_AND_ONNX_MODELS.md) — Detailed breakdown of Google LiteRT (`2.3.0`) / LiteRT-LM (`0.19.0`) and Microsoft ONNX Runtime (`1.31.1`) model selection, quantization trade-offs on 6 GB RAM, and direct Hugging Face / Google Cloud Storage download URLs.

## Key Architectural Highlights

1. **Dual First-Class On-Device Runtimes (LiteRT + ONNX Runtime)**:
   - **Google LiteRT (`com.google.ai.edge.litert:2.3.0`) & LiteRT-LM (`com.google.ai.edge.litertlm:litertlm-android:0.19.0`)**: Runs **Gemma 4 E2B IT (`gemma-4-E2B-it-gpu.litertlm`)** and **Gemma 3 1B IT** with OpenCL / ML Drift GPU acceleration on the 20-core Mali-G78 MP20 GPU, plus **Universal Sentence Encoder (`universal_sentence_encoder.tflite`)** via MediaPipe `TextEmbedder`.
   - **Microsoft ONNX Runtime (`com.microsoft.onnxruntime:onnxruntime-android:1.31.1`)**: Runs **Qwen2.5-1.5B-Instruct (INT4 Mobile)**, **all-MiniLM-L6-v2 (`model_qint8_arm64.onnx`, 384-D embeddings)**, and **Whisper Tiny (INT8 ASR)** using the ARM64 XNNPACK execution provider.
2. **All Four Capabilities Supported**:
   - **Text**: Streaming & non-streaming chat completions (`Gemma 4 E2B IT` on LiteRT & `Qwen2.5-1.5B-Instruct` on ONNX).
   - **Image / Vision**: Multimodal image understanding (`Content.ImageBytes` on LiteRT-LM and OpenAI `image_url` base64 payloads).
   - **Voice (ASR + TTS)**: Offline speech transcription via **Whisper Tiny INT8 ONNX** (`/v1/audio/transcriptions`) + hardware-accelerated **Pixel Tensor SpeechRecognizer & TextToSpeech** (`/v1/audio/speech`).
   - **Embeddings**: 384-dimensional L2-normalized semantic embeddings via `all-MiniLM-L6-v2` (ONNX) and 100-dimensional embeddings via `universal_sentence_encoder.tflite` (LiteRT) (`/v1/embeddings`).
3. **Zero Bundled Models + Post-Install Web Downloader**:
   - **0 bytes of model weights** are packaged inside the APK.
   - Built-in `ModelDownloadManager` streams models directly from Hugging Face and Google Cloud Storage over HTTPS with HTTP `Range` resume support, live progress bars, optional Hugging Face Bearer token authentication, and custom `.litertlm` / `.onnx` URL registration.
4. **Hardware & Android 17 (`DP11.261016.003`) Optimizations for Pixel 6a**:
   - **Tri-Cluster CPU Affinity Pinning (`tensor_affinity.cpp`)**: Pins the 4 inference worker threads to the Tensor GS101's 2x 2.80 GHz Cortex-X1 (`cpu6`–`cpu7`) and 2x 2.25 GHz Cortex-A76 (`cpu4`–`cpu5`) cores (`0xF0` mask) via `sched_setaffinity`, while pinning HTTP/disk I/O threads to the 4x 1.80 GHz Cortex-A55 cores (`0x0F` mask) so slow in-order cores never stall matrix multiplication barriers.
   - **Android Dynamic Performance Framework (`PerformanceHintManager`)**: Reports per-token step durations (`target = 25 ms`) to bypass the ~200 ms DVFS governor ramp latency on kernel `6.1.172-android14-11`.
   - **Read-Only `mmap` Weight Streaming (`RssFile` vs `RssAnon`)**: Maps model weights via `mmap(..., PROT_READ, MAP_SHARED, fd, 0)` and `madvise(MADV_SEQUENTIAL)` so multi-gigabyte weights stay in the reclaimable Linux file page cache (`RssFile`) instead of triggering Android 17's strict `MemoryLimiter:AnonSwap` OOM killer on 6 GB RAM.
   - **Android 17 Manifest & Linker Compliance**: Declares `<uses-permission android:name="android.permission.ACCESS_LOCAL_NETWORK" />` for LAN socket hosting, `<uses-native-library android:name="libOpenCL.so" />` + `libvndksupport.so` for Mali-G78 MP20 OpenCL access, 16 KB ELF page alignment (`-Wl,-z,max-page-size=16384`), and a persistent `connectedDevice|specialUse` Foreground Service.
5. **Ultra-Minimal Native Frontend (<5 MB RAM) + On-Device Test Console**:
   - Built with pure Android native Views (`Activity` + `LinearLayout`, zero Compose/AppCompat/WebView overhead).
   - Includes an **On-Device Direct Test Console** to test Text chat, Image vision attachments, Mic Voice dictation/TTS, and Embedding cosine similarity directly on the phone without needing a second device.

## OpenAI-Compatible Local Wi-Fi Endpoints

When the server is started in the app, connect any OpenAI SDK or client on the same Wi-Fi network to `http://<pixel6a-wifi-ip>:8080/v1` with header `Authorization: Bearer sk-pixel6a-tensor`:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Returns Pixel 6a hardware & Android 17 `RssAnon` / `RssFile` memory telemetry |
| `/v1/models` | `GET` | Lists LiteRT and ONNX models, installation status, and capabilities |
| `/v1/chat/completions` | `POST` | Chat & Vision completions (`stream: true` SSE `text/event-stream` or JSON) |
| `/v1/embeddings` | `POST` | Generates L2-normalized embedding vectors (`all-MiniLM-L6-v2` or `USE`) |
| `/v1/audio/transcriptions` | `POST` | Transcribes WAV/PCM audio via Whisper Tiny INT8 ONNX |
| `/v1/audio/speech` | `POST` | Synthesizes 16 kHz Mono RIFF WAV speech audio |
