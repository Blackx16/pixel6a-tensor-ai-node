plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.blackx16.tensorainode"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.blackx16.tensorainode"
        minSdk = 31
        // WHY: Compatible with Android 17 (API 37, build DP11.261016.003) on Pixel 6a while compiling cleanly on standard CI SDK 35.
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // WHY: Restrict ABI strictly to 64-bit arm64-v8a for the Pixel 6a's Google Tensor GS101.
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

    signingConfigs {
        create("release") {
            val ksFile = rootProject.file("keystore/release.jks")
            if (ksFile.exists()) {
                storeFile = ksFile
                storePassword = System.getenv("KEYSTORE_PASSWORD") ?: "pixel6a_tensor_pass"
                keyAlias = System.getenv("KEY_ALIAS") ?: "pixel6a_key"
                keyPassword = System.getenv("KEY_PASSWORD") ?: "pixel6a_tensor_pass"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isShrinkResources = false
            val ksFile = rootProject.file("keystore/release.jks")
            signingConfig = if (ksFile.exists()) {
                signingConfigs.getByName("release")
            } else {
                signingConfigs.getByName("debug")
            }
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf(
            "-opt-in=kotlinx.coroutines.ExperimentalCoroutinesApi"
        )
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }

    packaging {
        jniLibs {
            // WHY: Keep .so libraries uncompressed and 16 KB page-aligned as required by Android 15-17.
            useLegacyPackaging = false
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
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
    implementation(libs.mediapipe.tasks.genai)
    implementation(libs.mediapipe.tasks.text)
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    testImplementation(libs.org.json)
}
