package org.crc.multimodal.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Android ONNX Runtime inference engine for the frozen Colorectal Cancer Multimodal Fusion Model.
 *
 * Contract:
 * - Inputs:
 *   - "vision_seq": Shape [1, N, 1024], Type FLOAT (FP32)
 *   - "text_summary": Shape [1, 768], Type FLOAT (FP32)
 * - Outputs:
 *   - "logit": Shape [1, 1], Type FLOAT
 *   - "probability": Shape [1, 1], Type FLOAT
 *   - "attention_weights": Shape [1, 1, N], Type FLOAT
 * - Decision Rule:
 *   - probability >= 0.83f -> Class 1 (MSI)
 *   - probability < 0.83f  -> Class 0 (MSS)
 * - Evaluation Protocols:
 *   - PROTOCOL_1_CAPPED: 1 <= N <= 1000 patches (manifest-controlled).
 *   - PROTOCOL_2_UNCAPPED: Uncapped arbitrary patch counts (subject to hardware memory).
 * - Execution: Strictly single-patient sequential execution with deterministic native resource cleanup.
 */
class CrcMultimodalClassifier private constructor(
    private val env: OrtEnvironment,
    internal val session: OrtSession
) : AutoCloseable {

    companion object {
        const val INPUT_VISION_NAME = "vision_seq"
        const val INPUT_TEXT_NAME = "text_summary"
        const val OUTPUT_LOGIT_NAME = "logit"
        const val OUTPUT_PROBABILITY_NAME = "probability"
        const val OUTPUT_ATTENTION_NAME = "attention_weights"

        /**
         * Factory method to instantiate the classifier from a byte array (e.g., loaded from Android assets).
         */
        @JvmStatic
        fun fromByteArray(modelBytes: ByteArray, numThreads: Int = 2): CrcMultimodalClassifier {
            val env = OrtEnvironment.getEnvironment("CrcClassifierEnv")
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(numThreads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = env.createSession(modelBytes, sessionOptions)
            validateSessionModel(session)
            return CrcMultimodalClassifier(env, session)
        }

        /**
         * Factory method to instantiate the classifier from a local file path.
         */
        @JvmStatic
        fun fromFilePath(modelPath: String, numThreads: Int = 2): CrcMultimodalClassifier {
            val env = OrtEnvironment.getEnvironment("CrcClassifierEnv")
            val sessionOptions = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(numThreads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = env.createSession(modelPath, sessionOptions)
            validateSessionModel(session)
            return CrcMultimodalClassifier(env, session)
        }

        private fun validateSessionModel(session: OrtSession) {
            val inputNames = session.inputNames
            require(inputNames.contains(INPUT_VISION_NAME)) {
                "Model is missing required input: $INPUT_VISION_NAME (found: $inputNames)"
            }
            require(inputNames.contains(INPUT_TEXT_NAME)) {
                "Model is missing required input: $INPUT_TEXT_NAME (found: $inputNames)"
            }

            val outputNames = session.outputNames
            require(outputNames.contains(OUTPUT_LOGIT_NAME)) {
                "Model is missing required output: $OUTPUT_LOGIT_NAME (found: $outputNames)"
            }
            require(outputNames.contains(OUTPUT_PROBABILITY_NAME)) {
                "Model is missing required output: $OUTPUT_PROBABILITY_NAME (found: $outputNames)"
            }
            require(outputNames.contains(OUTPUT_ATTENTION_NAME)) {
                "Model is missing required output: $OUTPUT_ATTENTION_NAME (found: $outputNames)"
            }
        }
    }

    private val executionLock = Any()
    @Volatile
    private var isClosed = false

    /**
     * Executes single-patient multimodal fusion inference under an explicit protocol.
     *
     * @param visionData Flattened float array representing [1, numPatches, 1024].
     * @param numPatches Number of slide image patches (N).
     * @param textData Flattened float array representing [1, 768].
     * @param caseId Optional patient or deidentified case identifier.
     * @param protocol Explicit evaluation protocol (default: PROTOCOL_1_CAPPED).
     * @return InferenceResult containing probability, logit, predicted class, and attention weights.
     */
    fun predict(
        visionData: FloatArray,
        numPatches: Int,
        textData: FloatArray,
        caseId: String? = null,
        protocol: InferenceProtocol = InferenceProtocol.PROTOCOL_1_CAPPED
    ): InferenceResult {
        if (isClosed) {
            throw IllegalStateException("Classifier session is already closed.")
        }

        // 1. Strict input validation with explicit protocol policy
        InputValidator.validateVision(visionData, numPatches, protocol)
        InputValidator.validateText(textData)

        val startTime = System.currentTimeMillis()

        // 2. Synchronized execution to enforce single-case execution and thread safety
        synchronized(executionLock) {
            var visionTensor: OnnxTensor? = null
            var textTensor: OnnxTensor? = null
            var runResult: OrtSession.Result? = null

            try {
                // Direct NIO buffers with native byte order and integer overflow safeguards
                val visionBytesLong = visionData.size.toLong() * Float.SIZE_BYTES.toLong()
                require(visionBytesLong <= Int.MAX_VALUE.toLong()) {
                    "Vision buffer byte size ($visionBytesLong bytes) exceeds direct ByteBuffer capacity limit."
                }

                val textBytesLong = textData.size.toLong() * Float.SIZE_BYTES.toLong()
                require(textBytesLong <= Int.MAX_VALUE.toLong()) {
                    "Text buffer byte size ($textBytesLong bytes) exceeds direct ByteBuffer capacity limit."
                }

                val visionBuffer: FloatBuffer = ByteBuffer
                    .allocateDirect(visionBytesLong.toInt())
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                    .put(visionData)
                visionBuffer.flip()

                val textBuffer: FloatBuffer = ByteBuffer
                    .allocateDirect(textBytesLong.toInt())
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer()
                    .put(textData)
                textBuffer.flip()

                val visionShape = longArrayOf(1L, numPatches.toLong(), InputValidator.VISION_FEATURE_DIM.toLong())
                val textShape = longArrayOf(1L, InputValidator.TEXT_FEATURE_DIM.toLong())

                visionTensor = OnnxTensor.createTensor(env, visionBuffer, visionShape)
                textTensor = OnnxTensor.createTensor(env, textBuffer, textShape)

                val inputMap = mapOf(
                    INPUT_VISION_NAME to visionTensor,
                    INPUT_TEXT_NAME to textTensor
                )

                runResult = session.run(inputMap)

                // 3. Extract and parse model outputs supporting OnnxTensor natively
                val logitValue = extractScalarFloat(runResult.get(OUTPUT_LOGIT_NAME).get())
                val probabilityValue = extractScalarFloat(runResult.get(OUTPUT_PROBABILITY_NAME).get())
                val attentionArray = extractAttentionArray(runResult.get(OUTPUT_ATTENTION_NAME).get(), numPatches)

                // 4. Validate output finiteness and bounds
                if (logitValue.isNaN() || logitValue.isInfinite()) {
                    throw InferenceException("Model produced non-finite logit: $logitValue")
                }
                if (probabilityValue.isNaN() || probabilityValue.isInfinite()) {
                    throw InferenceException("Model produced non-finite probability: $probabilityValue")
                }
                if (probabilityValue !in 0.0f..1.0f) {
                    throw InferenceException("Model probability out of bounds [0.0, 1.0]: $probabilityValue")
                }
                if (attentionArray.size != numPatches) {
                    throw InferenceException("Attention output size mismatch: expected $numPatches, got ${attentionArray.size}")
                }
                for (i in attentionArray.indices) {
                    val a = attentionArray[i]
                    if (a.isNaN() || a.isInfinite() || a < 0.0f) {
                        throw InferenceException("Attention weight at index $i is invalid or negative: $a")
                    }
                }

                val latency = System.currentTimeMillis() - startTime
                val predictedClass = ThresholdLogic.classify(probabilityValue)
                val predictedLabel = ThresholdLogic.labelFor(predictedClass)

                return InferenceResult(
                    caseId = caseId,
                    predictedClass = predictedClass,
                    predictedLabel = predictedLabel,
                    probability = probabilityValue,
                    logit = logitValue,
                    threshold = ThresholdLogic.OPERATING_THRESHOLD,
                    numPatches = numPatches,
                    protocol = protocol,
                    attentionWeights = attentionArray,
                    latencyMs = latency
                )
            } catch (e: Exception) {
                if (e is IllegalArgumentException) throw e
                if (e is InferenceException) throw e
                throw InferenceException("Failed executing ONNX inference for case $caseId: ${e.message}", e)
            } finally {
                // 5. Deterministic native resource cleanup
                runResult?.close()
                visionTensor?.close()
                textTensor?.close()
            }
        }
    }

    /**
     * Backward-compatible overload for boolean enforceProtocol1Cap.
     */
    fun predict(
        visionData: FloatArray,
        numPatches: Int,
        textData: FloatArray,
        caseId: String? = null,
        enforceProtocol1Cap: Boolean
    ): InferenceResult {
        val protocol = if (enforceProtocol1Cap) {
            InferenceProtocol.PROTOCOL_1_CAPPED
        } else {
            InferenceProtocol.PROTOCOL_2_UNCAPPED
        }
        return predict(visionData, numPatches, textData, caseId, protocol)
    }

    private fun extractScalarFloat(rawOutput: Any): Float {
        return when (rawOutput) {
            is OnnxTensor -> rawOutput.floatBuffer.get(0)
            is FloatArray -> rawOutput[0]
            is FloatBuffer -> rawOutput.get(0)
            is Array<*> -> {
                val first = rawOutput[0]
                if (first is FloatArray) first[0]
                else if (first is Array<*>) (first[0] as Number).toFloat()
                else (first as Number).toFloat()
            }
            is Number -> rawOutput.toFloat()
            else -> throw InferenceException("Unexpected output tensor format for scalar: ${rawOutput.javaClass.name}")
        }
    }

    private fun extractAttentionArray(rawOutput: Any, expectedPatches: Int): FloatArray {
        return when (rawOutput) {
            is OnnxTensor -> {
                val fb = rawOutput.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                arr
            }
            is Array<*> -> {
                val b0 = rawOutput[0]
                if (b0 is Array<*>) {
                    val p0 = b0[0]
                    if (p0 is FloatArray) p0
                    else FloatArray(expectedPatches) { i -> ((p0 as Array<*>)[i] as Number).toFloat() }
                } else if (b0 is FloatArray) {
                    b0
                } else {
                    FloatArray(expectedPatches) { i -> (rawOutput[i] as Number).toFloat() }
                }
            }
            is FloatArray -> rawOutput
            is FloatBuffer -> {
                val arr = FloatArray(rawOutput.remaining())
                rawOutput.get(arr)
                arr
            }
            else -> throw InferenceException("Unexpected output tensor format for attention: ${rawOutput.javaClass.name}")
        }
    }

    override fun close() {
        synchronized(executionLock) {
            if (!isClosed) {
                isClosed = true
                session.close()
            }
        }
    }
}
