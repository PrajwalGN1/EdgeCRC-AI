package org.crc.multimodal.inference

import android.content.res.AssetManager
import android.util.Log
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Authentic holdout reference benchmark case data.
 */
data class ReferenceCaseData(
    val caseId: String,
    val numPatches: Int,
    val gtLabel: Int,
    val refLogit: Float,
    val refProb: Float,
    val refCall: Int,
    val textData: FloatArray,
    val visionData: FloatArray
)

/**
 * Loader for frozen authentic reference benchmark tensors packaged in APK assets.
 * Validates binary integrity, dimension bounds, and float finiteness prior to inference.
 */
object ReferenceTensorLoader {
    private const val TAG = "CRC_FP32_VERIFY"
    private val MAGIC = byteArrayOf('C'.code.toByte(), 'R'.code.toByte(), 'C'.code.toByte(), 'T'.code.toByte())

    private val CASE_ASSET_MAP = mapOf(
        "TCGA-D5-6539" to "reference_tensors/tcga_d5_6539.bin",  // N = 10
        "TCGA-AF-2693" to "reference_tensors/tcga_af_2693.bin",  // N = 52
        "TCGA-AG-A00C" to "reference_tensors/tcga_ag_a00c.bin",  // N = 99
        "TCGA-D5-6540" to "reference_tensors/tcga_d5_6540.bin",  // N = 345
        "TCGA-AA-3845" to "reference_tensors/tcga_aa_3845.bin",  // N = 502
        "TCGA-CM-6675" to "reference_tensors/tcga_cm_6675.bin",  // N = 541
        "TCGA-CM-6674" to "reference_tensors/tcga_cm_6674.bin"   // N = 1000
    )

    fun isReferenceCase(caseId: String): Boolean {
        return CASE_ASSET_MAP.containsKey(caseId.trim())
    }

    fun getReferenceAssetPath(caseId: String): String? {
        return CASE_ASSET_MAP[caseId.trim()]
    }

    fun loadFromAssets(assets: AssetManager, caseId: String): ReferenceCaseData {
        val assetPath = getReferenceAssetPath(caseId)
            ?: throw IllegalArgumentException("No authentic reference tensor registered for case: $caseId")
        
        CrcLogger.i("[TENSOR_LOAD_START] Loading authentic reference tensor: $caseId from $assetPath")
        return assets.open(assetPath).use { inputStream ->
            loadFromStream(inputStream, caseId)
        }
    }

    fun loadFromStream(inputStream: InputStream, expectedCaseId: String? = null): ReferenceCaseData {
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // 1. Verify Magic Header
        val magic = ByteArray(4)
        buffer.get(magic)
        if (!magic.contentEquals(MAGIC)) {
            val magicStr = String(magic, Charsets.US_ASCII)
            throw IllegalArgumentException("Invalid reference tensor magic header: expected CRCT, got $magicStr")
        }

        // 2. Version
        val version = buffer.int
        require(version == 1) { "Unsupported reference tensor version: $version (expected 1)" }

        // 3. Case Identifier
        val keyLen = buffer.int
        require(keyLen in 1..256) { "Invalid case ID length: $keyLen" }
        val keyBytes = ByteArray(keyLen)
        buffer.get(keyBytes)
        val caseId = String(keyBytes, Charsets.UTF_8)

        if (expectedCaseId != null && !caseId.equals(expectedCaseId.trim(), ignoreCase = true)) {
            throw IllegalArgumentException("Case ID mismatch in reference tensor: expected $expectedCaseId, got $caseId")
        }

        // 4. Metadata & Reference Targets
        val numPatches = buffer.int
        val gtLabel = buffer.int
        val refLogit = buffer.float
        val refProb = buffer.float
        val refCall = buffer.int

        // 5. Text Embedding: 768 floats
        val textData = FloatArray(InputValidator.TEXT_FEATURE_DIM)
        val textFloatBuf = buffer.asFloatBuffer()
        textFloatBuf.get(textData)
        buffer.position(buffer.position() + InputValidator.TEXT_FEATURE_DIM * Float.SIZE_BYTES)

        // 6. Vision Sequence: numPatches * 1024 floats
        val visionFloatsCount = numPatches * InputValidator.VISION_FEATURE_DIM
        val visionData = FloatArray(visionFloatsCount)
        val visionFloatBuf = buffer.asFloatBuffer()
        visionFloatBuf.get(visionData)
        buffer.position(buffer.position() + visionFloatsCount * Float.SIZE_BYTES)

        // 7. Validate Tensors
        InputValidator.validateText(textData)
        InputValidator.validateVision(visionData, numPatches, InferenceProtocol.PROTOCOL_1_CAPPED)

        CrcLogger.i(
            "[TENSOR_LOAD_SUCCESS] Case: $caseId | Patches: $numPatches | GT: $gtLabel | " +
            "RefLogit: ${String.format(java.util.Locale.US, "%.5f", refLogit)} | " +
            "RefProb: ${String.format(java.util.Locale.US, "%.5f", refProb)} | " +
            "RefCall: $refCall (${if (refCall == 1) "MSI" else "MSS"}) | " +
            "TextFloats: ${textData.size} | VisionFloats: ${visionData.size}"
        )

        return ReferenceCaseData(
            caseId = caseId,
            numPatches = numPatches,
            gtLabel = gtLabel,
            refLogit = refLogit,
            refProb = refProb,
            refCall = refCall,
            textData = textData,
            visionData = visionData
        )
    }
}
