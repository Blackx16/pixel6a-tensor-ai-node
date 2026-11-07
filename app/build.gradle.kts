plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.blackx16.tensorainode"
    // WHY: Target Android 17 (API level 37, build DP11.261016.003) running on the user's Pixel 6a.
    compileSdk = 37

    defaultConfig {
        applicationId = "com.blackx16.tensorainode"
        minSdk = 31
        targetSdk = 37
        versionCode = 1
        versionName = "1.0.0-pixel6a-a17"

        // WHY: Restrict ABI strictly to 64-bit arm64-v8a for the Pixel 6a's Google Tensor GS101
        // to avoid packaging unused x86_64 or armeabi-v7a native binaries.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                // WHY: Enable ARMv8.2-A dot-product (+dotprod) and FP16 (+fp16) instructions
                // supported by Tensor GS101's Cortex-X1, Cortex-A76, and Cortex-A55 cores.
                cppFlags += listOf(
                    "-std=c++17",
                    "-O3",
                    "-march=armv8.2-a+dotprod+fp16",
                    "-ffast-math"
                )
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON"
                )
            }
        }
    }

    buildTypes {
        release {
            // WHY: Full R8 minification & resource shrinking strip unused classes so the frontend
            // footprint stays under a few megabytes of RAM.
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        // WHY: LiteRT 2.3.0 and LiteRT-LM 0.19.0 ship with Kotlin 2.4 metadata; skipping the metadata
        // version check ensures smooth compilation across Kotlin 2.2/2.3 toolchains.
        freeCompilerArgs += listOf(
            "-Xskip-metadata-version-check",
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi"
        )
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs {
            // WHY: Keep .so libraries uncompressed and 16 KB page-aligned as required by Android 15-17.
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.coroutines.android)

    // WHY: Include both Google LiteRT ("Lightartie") and Microsoft ONNX Runtime ("Onyx")
    // so both model ecosystems run natively with hardware acceleration.
    implementation(libs.litert.core)
    implementation(libs.litert.gpu)
    implementation(libs.litert.support)
    implementation(libs.litertlm.android)
    implementation(libs.mediapipe.tasks.genai)
    implementation(libs.mediapipe.tasks.text)
    implementation(libs.onnxruntime.android)
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))

    testImplementation(libs.junit)
}
