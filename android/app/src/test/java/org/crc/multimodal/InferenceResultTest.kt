package org.crc.multimodal

import org.crc.multimodal.inference.InferenceResult
import org.junit.Assert.fail
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class InferenceResultTest {

    @Test
    fun testValidInferenceResultConstruction() {
        val res = InferenceResult(
            caseId = "CASE_36",
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.8301065f,
            logit = 1.586f,
            threshold = 0.83f,
            numPatches = 80,
            attentionWeights = FloatArray(80) { 1.0f / 80f },
            latencyMs = 15L
        )

        assertEquals("CASE_36", res.caseId)
        assertEquals(1, res.predictedClass)
        assertEquals("MSI", res.predictedLabel)
        assertEquals(0.8301065f, res.probability)
        assertEquals(80, res.numPatches)
        assertNotNull(res.attentionWeights)
        assertEquals(80, res.attentionWeights!!.size)
        assertTrue(res.disclaimer.contains("RESEARCH USE ONLY"))
    }

    @Test
    fun testInvalidClassThrows() {
        try {
            InferenceResult(
                predictedClass = 2, // invalid
                predictedLabel = "UNKNOWN",
                probability = 0.5f,
                logit = 0.0f,
                numPatches = 10
            )
            fail("Expected exception for invalid class")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("predictedClass"))
        }
    }

    @Test
    fun testInvalidProbabilityThrows() {
        try {
            InferenceResult(
                predictedClass = 0,
                predictedLabel = "MSS",
                probability = 1.2f, // invalid
                logit = 2.0f,
                numPatches = 10
            )
            fail("Expected exception for probability > 1.0")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("probability"))
        }
    }
}
