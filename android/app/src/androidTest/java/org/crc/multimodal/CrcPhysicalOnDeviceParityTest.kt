package org.crc.multimodal

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ReferenceTensorLoader
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import java.security.MessageDigest
import kotlin.math.abs

/**
 * Physical Android Device (Samsung Galaxy S24) On-Device FP32 Parity Verification Test.
 *
 * Runs natively on the Android OS runtime via connected AndroidTest.
 * Evaluates the frozen models/fusion_model_fp32.onnx on real mobile ARM64 hardware
 * using authentic paired multimodal tensors (TCGA-D5-6540 and TCGA-CM-6675).
 */
@RunWith(AndroidJUnit4::class)
class CrcPhysicalOnDeviceParityTest {

    companion object {
        private const val EXPECTED_MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        private const val ACCEPTANCE_TOLERANCE = 1.0e-4f
        private var classifier: CrcMultimodalClassifier? = null

        @BeforeClass
        @JvmStatic
        fun setUpDevice() {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val assets = context.assets

            val assetPath = "models/fusion_model_fp32.onnx"
            val modelBytes = assets.open(assetPath).use { it.readBytes() }

            // Verify model integrity on device
            val digest = MessageDigest.getInstance("SHA-256")
            val sha256 = digest.digest(modelBytes).joinToString("") { "%02x".format(it) }

            CrcLogger.i("[PHYSICAL_DEVICE_SETUP] Device Model: ${android.os.Build.MODEL} | Android: ${android.os.Build.VERSION.RELEASE} | ABI: ${android.os.Build.SUPPORTED_ABIS.joinToString()}")
            CrcLogger.i("[PHYSICAL_DEVICE_MODEL] Asset: $assetPath | Bytes: ${modelBytes.size} | SHA-256: $sha256")

            assertEquals(
                "Model SHA-256 on physical device must match frozen FP32 baseline",
                EXPECTED_MODEL_SHA256,
                sha256.lowercase()
            )

            classifier = CrcMultimodalClassifier.fromByteArray(modelBytes, numThreads = 2)
            CrcLogger.i("[PHYSICAL_DEVICE_INIT] CrcMultimodalClassifier initialized successfully on physical device.")
        }

        @AfterClass
        @JvmStatic
        fun tearDownDevice() {
            classifier?.close()
            classifier = null
            CrcLogger.i("[PHYSICAL_DEVICE_TEARDOWN] Classifier closed.")
        }
    }

    @Test
    fun verifyPhysicalOnDeviceMsiParityTcgaD56540() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val caseData = ReferenceTensorLoader.loadFromAssets(context.assets, "TCGA-D5-6540")

        val startMs = System.currentTimeMillis()
        val result = classifier!!.predict(
            visionData = caseData.visionData,
            numPatches = caseData.numPatches,
            textData = caseData.textData,
            caseId = caseData.caseId,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val latencyMs = System.currentTimeMillis() - startMs

        val logitDiff = abs(result.logit - caseData.refLogit)
        val probDiff = abs(result.probability - caseData.refProb)
        val parityPass = (probDiff <= ACCEPTANCE_TOLERANCE) && (result.predictedClass == caseData.refCall)

        CrcLogger.i(
            "[PHYSICAL_DEVICE_PARITY_MSI] Case: ${caseData.caseId} | Patches: ${caseData.numPatches} | " +
            "DeviceLogit: ${String.format(java.util.Locale.US, "%.5f", result.logit)} | RefLogit: ${String.format(java.util.Locale.US, "%.5f", caseData.refLogit)} | DiffLogit: ${String.format(java.util.Locale.US, "%.2e", logitDiff)} | " +
            "DeviceProb: ${String.format(java.util.Locale.US, "%.5f", result.probability)} | RefProb: ${String.format(java.util.Locale.US, "%.5f", caseData.refProb)} | DiffProb: ${String.format(java.util.Locale.US, "%.2e", probDiff)} | " +
            "Class: ${result.predictedClass} (${result.predictedLabel}) | RefCall: ${caseData.refCall} | " +
            "Latency: ${latencyMs}ms | ParityGate: ${if (parityPass) "PASS" else "FAIL"}"
        )

        assertEquals("Predicted class on physical device must be MSI (1)", 1, result.predictedClass)
        assertEquals("Predicted label on physical device must be MSI", "MSI", result.predictedLabel)
        assertTrue("Logit difference ($logitDiff) must be <= $ACCEPTANCE_TOLERANCE", logitDiff <= ACCEPTANCE_TOLERANCE)
        assertTrue("Probability difference ($probDiff) must be <= $ACCEPTANCE_TOLERANCE", probDiff <= ACCEPTANCE_TOLERANCE)
    }

    @Test
    fun verifyPhysicalOnDeviceMssParityTcgaCm6675() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val caseData = ReferenceTensorLoader.loadFromAssets(context.assets, "TCGA-CM-6675")

        val startMs = System.currentTimeMillis()
        val result = classifier!!.predict(
            visionData = caseData.visionData,
            numPatches = caseData.numPatches,
            textData = caseData.textData,
            caseId = caseData.caseId,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val latencyMs = System.currentTimeMillis() - startMs

        val logitDiff = abs(result.logit - caseData.refLogit)
        val probDiff = abs(result.probability - caseData.refProb)
        val parityPass = (probDiff <= ACCEPTANCE_TOLERANCE) && (result.predictedClass == caseData.refCall)

        CrcLogger.i(
            "[PHYSICAL_DEVICE_PARITY_MSS] Case: ${caseData.caseId} | Patches: ${caseData.numPatches} | " +
            "DeviceLogit: ${String.format(java.util.Locale.US, "%.5f", result.logit)} | RefLogit: ${String.format(java.util.Locale.US, "%.5f", caseData.refLogit)} | DiffLogit: ${String.format(java.util.Locale.US, "%.2e", logitDiff)} | " +
            "DeviceProb: ${String.format(java.util.Locale.US, "%.5f", result.probability)} | RefProb: ${String.format(java.util.Locale.US, "%.5f", caseData.refProb)} | DiffProb: ${String.format(java.util.Locale.US, "%.2e", probDiff)} | " +
            "Class: ${result.predictedClass} (${result.predictedLabel}) | RefCall: ${caseData.refCall} | " +
            "Latency: ${latencyMs}ms | ParityGate: ${if (parityPass) "PASS" else "FAIL"}"
        )

        assertEquals("Predicted class on physical device must be MSS (0)", 0, result.predictedClass)
        assertEquals("Predicted label on physical device must be MSS", "MSS", result.predictedLabel)
        assertTrue("Logit difference ($logitDiff) must be <= $ACCEPTANCE_TOLERANCE", logitDiff <= ACCEPTANCE_TOLERANCE)
        assertTrue("Probability difference ($probDiff) must be <= $ACCEPTANCE_TOLERANCE", probDiff <= ACCEPTANCE_TOLERANCE)
    }

    @Test
    fun verifyPhysicalOnDeviceAuthenticBundleImportTcgaA63807() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val bundleDir = java.io.File(context.filesDir, "specimen_bundles")
        bundleDir.mkdirs()
        val bundleFile = java.io.File(bundleDir, "TCGA-A6-3807.crcbundle")

        if (!bundleFile.exists()) {
            context.assets.open("bundles/TCGA-A6-3807.crcbundle").use { input ->
                bundleFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }
        assertTrue("Bundle file must exist in app-private storage: ${bundleFile.absolutePath}", bundleFile.exists())

        val bundle = org.crc.multimodal.inference.AuthenticBundleLoader.loadFromFile(bundleFile)

        assertEquals("TCGA-A6-3807", bundle.specimenId)
        assertEquals(613, bundle.patchCount)

        val startMs = System.currentTimeMillis()
        val result = classifier!!.predictAuthentic(
            visionData = bundle.visionData,
            numPatches = bundle.patchCount,
            textData = bundle.textData,
            caseId = bundle.specimenId,
            provenance = org.crc.multimodal.inference.InputProvenance.AUTHENTIC_IMPORTED
        )
        val latencyMs = System.currentTimeMillis() - startMs

        val refLogit = -6.061559f
        val refProb = 0.002325f
        val logitDiff = abs(result.logit - refLogit)
        val probDiff = abs(result.probability - refProb)

        CrcLogger.i(
            "[PHYSICAL_DEVICE_AUTHENTIC_BUNDLE] Case: ${bundle.specimenId} | Patches: ${bundle.patchCount} | " +
            "DeviceLogit: ${String.format(java.util.Locale.US, "%.5f", result.logit)} | RefLogit: ${String.format(java.util.Locale.US, "%.5f", refLogit)} | DiffLogit: ${String.format(java.util.Locale.US, "%.2e", logitDiff)} | " +
            "DeviceProb: ${String.format(java.util.Locale.US, "%.5f", result.probability)} | RefProb: ${String.format(java.util.Locale.US, "%.5f", refProb)} | DiffProb: ${String.format(java.util.Locale.US, "%.2e", probDiff)} | " +
            "Class: ${result.predictedClass} (${result.predictedLabel}) | Provenance: ${result.provenance} | " +
            "Latency: ${latencyMs}ms"
        )

        assertEquals("Predicted class must be MSS (0)", 0, result.predictedClass)
        assertEquals("Predicted label must be MSS", "MSS", result.predictedLabel)
        assertEquals(org.crc.multimodal.inference.InputProvenance.AUTHENTIC_IMPORTED, result.provenance)
        assertTrue("Logit difference ($logitDiff) must be <= $ACCEPTANCE_TOLERANCE", logitDiff <= ACCEPTANCE_TOLERANCE)
        assertTrue("Probability difference ($probDiff) must be <= $ACCEPTANCE_TOLERANCE", probDiff <= ACCEPTANCE_TOLERANCE)
    }

    @Test
    fun verifyPhysicalOnDeviceSyntheticProvenanceFailsClosed() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val bundleDir = java.io.File(context.filesDir, "specimen_bundles")
        bundleDir.mkdirs()
        val bundleFile = java.io.File(bundleDir, "TCGA-A6-3807.crcbundle")

        if (!bundleFile.exists()) {
            context.assets.open("bundles/TCGA-A6-3807.crcbundle").use { input ->
                bundleFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        val bundle = org.crc.multimodal.inference.AuthenticBundleLoader.loadFromFile(bundleFile)

        var exceptionThrown = false
        try {
            classifier!!.predictAuthentic(
                visionData = bundle.visionData,
                numPatches = bundle.patchCount,
                textData = bundle.textData,
                caseId = bundle.specimenId,
                provenance = org.crc.multimodal.inference.InputProvenance.DEMONSTRATION_SIMULATED
            )
        } catch (e: IllegalArgumentException) {
            exceptionThrown = true
            assertTrue(
                "Error message must specify authentic embeddings required",
                e.message?.contains("Authentic feature embeddings are required") == true
            )
        }
        assertTrue("predictAuthentic must fail closed with IllegalArgumentException on simulated provenance", exceptionThrown)
    }
}
