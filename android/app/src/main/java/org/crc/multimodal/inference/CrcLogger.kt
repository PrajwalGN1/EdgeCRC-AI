package org.crc.multimodal.inference

import android.util.Log
import org.crc.multimodal.security.SensitiveLogRedactor

/**
 * Diagnostic logger specifically scoped for FP32 On-Device Inference Verification.
 * Logs structured diagnostic information under TAG "CRC_FP32_VERIFY".
 * Automatically redacts clinical identifiers and PII using SensitiveLogRedactor.
 * Provides JVM-safe fallback for unit tests.
 */
object CrcLogger {
    const val TAG = "CRC_FP32_VERIFY"

    fun i(message: String, tag: String = TAG) {
        val safeMessage = SensitiveLogRedactor.redact(message)
        try {
            Log.i(tag, safeMessage)
        } catch (t: Throwable) {
            println("[$tag] INFO: $safeMessage")
        }
    }

    fun d(message: String, tag: String = TAG) {
        val safeMessage = SensitiveLogRedactor.redact(message)
        try {
            Log.d(tag, safeMessage)
        } catch (t: Throwable) {
            println("[$tag] DEBUG: $safeMessage")
        }
    }

    fun w(message: String, tag: String = TAG) {
        val safeMessage = SensitiveLogRedactor.redact(message)
        try {
            Log.w(tag, safeMessage)
        } catch (t: Throwable) {
            println("[$tag] WARN: $safeMessage")
        }
    }

    fun e(message: String, throwable: Throwable? = null, tag: String = TAG) {
        val safeMessage = SensitiveLogRedactor.redact(message)
        try {
            if (throwable != null) {
                Log.e(tag, safeMessage, throwable)
            } else {
                Log.e(tag, safeMessage)
            }
        } catch (t: Throwable) {
            println("[$tag] ERROR: $safeMessage")
            throwable?.printStackTrace()
        }
    }
}
