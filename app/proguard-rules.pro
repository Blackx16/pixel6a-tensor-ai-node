# WHY: Prevent R8 from stripping JNI entrypoints and protobuf message classes used by
# LiteRT, LiteRT-LM, MediaPipe Tasks, and ONNX Runtime (avoids SIGSEGV in native initialization).
-keep class com.google.ai.edge.litert.** { *; }
-keep class com.google.ai.edge.litertlm.** { *; }
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-keep class ai.onnxruntime.** { *; }

-keepclasseswithmembernames class com.blackx16.tensorainode.hardware.TensorNativeBridge {
    native <methods>;
}
