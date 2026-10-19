package org.crc.multimodal

import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputValidator
import org.junit.Assert.fail
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class InputValidatorTest {

    @Test
    fun testValidVisionEmbeddingPassesProtocol1() {
        val numPatches = 5
        val validVision = FloatArray(numPatches * 1024) { 0.1f }
        InputValidator.validateVision(validVision, numPatches, protocol = InferenceProtocol.PROTOCOL_1_CAPPED)
    }

    @Test
    fun testValidVisionEmbeddingPassesProtocol2() {
        val numPatches = 5
        val validVision = FloatArray(numPatches * 1024) { 0.1f }
        InputValidator.validateVision(validVision, numPatches, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)
    }

    @Test
    fun testValidTextEmbeddingPasses() {
        val validText = FloatArray(768) { -0.05f }
        InputValidator.validateText(validText)
    }

    @Test
    fun testZeroPatchesRejected() {
        try {
            InputValidator.validateVision(FloatArray(0), 0)
            fail("Expected IllegalArgumentException for 0 patches")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("positive"))
        }
    }

    @Test
    fun testNegativePatchesRejected() {
        try {
            InputValidator.validateVision(FloatArray(1024), -1)
            fail("Expected IllegalArgumentException for negative patches")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("positive"))
        }
    }

    @Test
    fun testProtocol1RejectsAbove1000Patches() {
        val numPatches = 1001
        val largeVision = FloatArray(numPatches * 1024)
        try {
            InputValidator.validateVision(largeVision, numPatches, protocol = InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected IllegalArgumentException for Protocol 1 exceeding 1000 patches")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Protocol 1 enforces a maximum cap of 1000 patches"))
        }
    }

    @Test
    fun testProtocol2AcceptsAbove1000Patches() {
        val numPatches = 1001
        val largeVision = FloatArray(numPatches * 1024)
        // Under Protocol 2, 1001 patches must pass cleanly without rejection
        InputValidator.validateVision(largeVision, numPatches, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)
    }

    @Test
    fun testProtocol2AcceptsCase20LargePatchCount() {
        // CASE_20 in holdout cohort contains exactly 3,914 patches under Protocol 2
        val numPatches = 3914
        val case20Vision = FloatArray(numPatches * 1024) { 0.01f }
        InputValidator.validateVision(case20Vision, numPatches, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)
    }

    @Test
    fun testUncappedInputNotRejectedMerelyBecauseAboveProtocol1Limit() {
        // Explicit check: ensure an uncapped input (e.g. 1500 patches) is not rejected merely because N > 1000
        val numPatches = 1500
        val visionBuffer = FloatArray(numPatches * 1024)
        InputValidator.validateVision(visionBuffer, numPatches, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)
        InputValidator.validateVision(visionBuffer, numPatches, enforceProtocol1Cap = false)
    }

    @Test
    fun testBackwardCompatibleBooleanOverload() {
        val numPatches = 1001
        val buffer = FloatArray(numPatches * 1024)
        
        // enforceProtocol1Cap = true should reject
        try {
            InputValidator.validateVision(buffer, numPatches, enforceProtocol1Cap = true)
            fail("Expected IllegalArgumentException with enforceProtocol1Cap = true")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Protocol 1 enforces a maximum cap of 1000 patches"))
        }

        // enforceProtocol1Cap = false should accept
        InputValidator.validateVision(buffer, numPatches, enforceProtocol1Cap = false)
    }

    @Test
    fun testVisionBufferLengthMismatchRejected() {
        val numPatches = 10
        val shortBuffer = FloatArray(numPatches * 1024 - 1)
        try {
            InputValidator.validateVision(shortBuffer, numPatches)
            fail("Expected IllegalArgumentException for mismatched buffer size")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("mismatch"))
        }
    }

    @Test
    fun testTextBufferLengthMismatchRejected() {
        val wrongText = FloatArray(512)
        try {
            InputValidator.validateText(wrongText)
            fail("Expected IllegalArgumentException for text dimension != 768")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("768"))
        }
    }

    @Test
    fun testNaNInVisionEmbeddingRejected() {
        val numPatches = 2
        val buffer = FloatArray(numPatches * 1024) { 0.0f }
        buffer[500] = Float.NaN
        try {
            InputValidator.validateVision(buffer, numPatches)
            fail("Expected IllegalArgumentException for NaN")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("NaN"))
        }
    }

    @Test
    fun testInfinityInTextEmbeddingRejected() {
        val buffer = FloatArray(768) { 0.0f }
        buffer[10] = Float.POSITIVE_INFINITY
        try {
            InputValidator.validateText(buffer)
            fail("Expected IllegalArgumentException for +Infinity")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("Infinite"))
        }
    }

    @Test
    fun testIntegerOverflowPatchCountRejected() {
        // N = 2,500,000 patches would require 2,500,000 * 1024 = 2,560,000,000 floats (> Int.MAX_VALUE)
        try {
            InputValidator.validateVision(FloatArray(0), 2_500_000, protocol = InferenceProtocol.PROTOCOL_2_UNCAPPED)
            fail("Expected IllegalArgumentException for integer overflow in patch count")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("exceeds memory allocation capacity limit"))
        }
    }
}
