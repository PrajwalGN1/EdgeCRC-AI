package org.crc.multimodal

import org.crc.multimodal.inference.ThresholdLogic
import org.junit.Assert.fail
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ThresholdLogicTest {

    @Test
    fun testExactThresholdBoundarySemantics() {
        val tau = 0.83f

        // Probability immediately below 0.83 -> Class 0 (MSS)
        val belowTau = 0.829999f
        assertEquals(0, ThresholdLogic.classify(belowTau, tau))
        assertEquals("MSS", ThresholdLogic.labelFor(ThresholdLogic.classify(belowTau, tau)))

        // Probability exactly at 0.83 -> Class 1 (MSI)
        val exactTau = 0.830000f
        assertEquals(1, ThresholdLogic.classify(exactTau, tau))
        assertEquals("MSI", ThresholdLogic.labelFor(ThresholdLogic.classify(exactTau, tau)))

        // Probability immediately above 0.83 -> Class 1 (MSI)
        val aboveTau = 0.830001f
        assertEquals(1, ThresholdLogic.classify(aboveTau, tau))
        assertEquals("MSI", ThresholdLogic.labelFor(ThresholdLogic.classify(aboveTau, tau)))
    }

    @Test
    fun testExtremesAndTypicalCases() {
        // High confidence MSI
        assertEquals(1, ThresholdLogic.classify(0.999f))
        assertEquals("MSI", ThresholdLogic.labelFor(1))

        // High confidence MSS
        assertEquals(0, ThresholdLogic.classify(0.001f))
        assertEquals("MSS", ThresholdLogic.labelFor(0))

        // Edge 0.0 and 1.0
        assertEquals(0, ThresholdLogic.classify(0.0f))
        assertEquals(1, ThresholdLogic.classify(1.0f))
    }

    @Test
    fun testDistanceCalculations() {
        val tau = 0.83f
        val prob = 0.85f
        assertEquals(0.02f, ThresholdLogic.signedDistanceToThreshold(prob, tau), 1e-6f)
        assertEquals(0.02f, ThresholdLogic.marginToThreshold(prob, tau), 1e-6f)

        val probBelow = 0.80f
        assertEquals(-0.03f, ThresholdLogic.signedDistanceToThreshold(probBelow, tau), 1e-6f)
        assertEquals(0.03f, ThresholdLogic.marginToThreshold(probBelow, tau), 1e-6f)
    }

    @Test
    fun testInvalidProbabilitiesRejected() {
        try {
            ThresholdLogic.classify(-0.01f)
            fail("Expected exception for probability < 0.0")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("[0.0, 1.0]"))
        }

        try {
            ThresholdLogic.classify(1.05f)
            fail("Expected exception for probability > 1.0")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("[0.0, 1.0]"))
        }

        try {
            ThresholdLogic.classify(Float.NaN)
            fail("Expected exception for NaN")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("NaN"))
        }
    }
}
