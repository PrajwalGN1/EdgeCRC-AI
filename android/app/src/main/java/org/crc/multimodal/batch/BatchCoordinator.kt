package org.crc.multimodal.batch

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.SpecimenInput
import org.crc.multimodal.data.WorkstationRepository
import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.BundleStorageManager
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputProvenance
import org.crc.multimodal.inference.InputValidator
import java.io.File
import java.security.MessageDigest
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Coordinates multi-specimen batch ingestion, deterministic execution, and state transitions.
 */
class BatchCoordinator(
    private val repository: BatchRepository,
    private val workstationRepository: WorkstationRepository? = null
) {

    companion object {
        const val MAX_BATCH_SIZE = 100
        const val MODEL_SHA256 = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"

        fun generateBatchId(): String {
            val dateStr = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
            val suffix = UUID.randomUUID().toString().take(6).uppercase(Locale.US)
            return "BATCH-$dateStr-$suffix"
        }

        /**
         * Resolves the appropriate BatchRun to display based on navigation arguments:
         * - If [requestedBatchId] is provided, attempts to load that specific batch. If not found, returns null (never substitutes).
         * - If [requestedBatchId] is null or blank, retrieves the latest batch ordered deterministically (created_at DESC).
         * - If repository has no batches or an error occurs, returns null.
         */
        fun resolveBatchForDisplay(repository: BatchRepository, requestedBatchId: String?): BatchRun? {
            return try {
                if (!requestedBatchId.isNullOrBlank()) {
                    val batch = repository.getBatchRun(requestedBatchId)
                    if (batch == null) {
                        CrcLogger.w("[BATCH_RESTORE] Explicit batch '$requestedBatchId' not found. Preserving empty state.")
                    }
                    batch
                } else {
                    repository.getAllBatchRuns().firstOrNull()
                }
            } catch (e: Exception) {
                CrcLogger.e("[BATCH_RESTORE] Error resolving batch for display: ${e.message}", e)
                null
            }
        }

        /**
         * Persists an eligible completed specimen result to the shared Cases repository.
         * Ensures idempotency: will not overwrite or duplicate existing validated records.
         */
        fun persistCompletedSpecimenToCases(
            context: Context? = null,
            item: BatchSpecimenItem,
            workstationRepo: WorkstationRepository?
        ) {
            if (workstationRepo == null) return
            if (item.status != BatchSpecimenStatus.COMPLETED || item.probability == null) return

            val specimenId = item.specimenId.trim()
            if (specimenId.isBlank()) return

            try {
                val existingCase = workstationRepo.getCaseByIdentifier(specimenId)
                val targetCaseId: Long
                val targetProtocol = InferenceProtocol.fromStringOrNull(item.protocol) ?: InferenceProtocol.PROTOCOL_1_CAPPED

                if (existingCase == null) {
                    val newCase = ResearchCase(
                        caseIdentifier = specimenId,
                        specimenDescription = "Batch specimen (${item.fileName.ifBlank { specimenId }})",
                        researchContext = "Originating Batch: ${item.batchId}\nSource: Batch Cohort Ingestion",
                        status = CaseStatus.ANALYZED,
                        patchCount = item.patchCount.coerceAtLeast(1),
                        protocol = targetProtocol,
                        createdAt = item.completedTimestamp ?: System.currentTimeMillis(),
                        updatedAt = item.completedTimestamp ?: System.currentTimeMillis()
                    )
                    targetCaseId = workstationRepo.insertCase(newCase)

                    val bundleFile = if (context != null) BundleStorageManager.getBundleFile(context, specimenId) else null
                    try {
                        workstationRepo.insertSpecimenInput(
                            SpecimenInput(
                                caseId = targetCaseId,
                                inputType = "AUTHENTIC_BUNDLE",
                                localUriOrAsset = bundleFile?.absolutePath ?: item.sourceUri,
                                filename = item.fileName.ifBlank { "$specimenId.crcbundle" },
                                validationStatus = "VERIFIED_AUTHENTIC",
                                createdAt = item.completedTimestamp ?: System.currentTimeMillis()
                            )
                        )
                    } catch (e: Exception) {
                        CrcLogger.w("[BATCH_CASES_INTEGRATION] Non-critical warning inserting specimen input: ${e.message}")
                    }
                } else {
                    targetCaseId = existingCase.id
                    if (existingCase.status == CaseStatus.ANALYZED ||
                        existingCase.status == CaseStatus.UNDER_REVIEW ||
                        existingCase.status == CaseStatus.REVIEWED
                    ) {
                        CrcLogger.i("[BATCH_CASES_INTEGRATION] Specimen '$specimenId' already exists in Cases (status=${existingCase.status.name}). Preserving existing record.")
                        return
                    }
                }

                val inferenceRun = InferenceRun(
                    caseId = targetCaseId,
                    modelIdentifier = "fusion_model_fp32.onnx",
                    inferenceArchitecture = "LOCAL_ONNX_FP32",
                    protocol = targetProtocol,
                    predictedClass = item.predictedClass ?: 0,
                    predictedLabel = item.predictedLabel ?: if (item.predictedClass == 1) "MSI" else "MSS",
                    probability = item.probability,
                    logit = item.logit ?: 0f,
                    threshold = item.threshold,
                    margin = item.probability - item.threshold,
                    latencyMs = item.inferenceLatencyMs.toDouble(),
                    numPatches = item.patchCount.coerceAtLeast(1),
                    diagnosticsSummary = "Batch execution in ${item.batchId}. Provenance: ${item.provenance.name}.",
                    executionTimestamp = item.completedTimestamp ?: System.currentTimeMillis(),
                    status = "SUCCESS"
                )
                workstationRepo.recordInferenceResult(
                    caseId = targetCaseId,
                    run = inferenceRun,
                    newStatus = CaseStatus.ANALYZED
                )
                workstationRepo.recordAuditEvent(
                    org.crc.multimodal.security.AuditTrailRecord(
                        actionType = org.crc.multimodal.security.AuditActionType.INFERENCE_EXECUTED,
                        specimenId = specimenId,
                        details = "Batch inference registered for specimen '$specimenId' in batch '${item.batchId}'. Predicted: ${inferenceRun.predictedLabel} (P=${item.probability})"
                    )
                )
                CrcLogger.i("[BATCH_CASES_INTEGRATION] Successfully registered batch specimen '$specimenId' into Cases registry (Case ID: $targetCaseId).")
            } catch (e: Exception) {
                CrcLogger.e("[BATCH_CASES_INTEGRATION] Error persisting specimen '$specimenId' to Cases: ${e.message}", e)
            }
        }

        /**
         * Synchronizes all active completed specimens from a batch run to the shared Cases repository.
         * Safe, idempotent, and non-destructive. Returns count of newly persisted cases.
         */
        fun syncBatchToCases(
            context: Context? = null,
            batchRun: BatchRun,
            workstationRepo: WorkstationRepository?
        ): Int {
            if (workstationRepo == null) return 0
            val completedItems = batchRun.items.filter {
                it.status == BatchSpecimenStatus.COMPLETED && it.probability != null && !it.isArchived
            }
            var newlySynced = 0
            for (item in completedItems) {
                val existing = workstationRepo.getCaseByIdentifier(item.specimenId)
                if (existing == null) {
                    persistCompletedSpecimenToCases(context, item, workstationRepo)
                    newlySynced++
                } else if (existing.status != CaseStatus.ANALYZED &&
                    existing.status != CaseStatus.UNDER_REVIEW &&
                    existing.status != CaseStatus.REVIEWED
                ) {
                    persistCompletedSpecimenToCases(context, item, workstationRepo)
                    newlySynced++
                }
            }
            return newlySynced
        }
    }

    /**
     * Ingests and validates multiple authentic bundles selected via SAF.
     */
    fun ingestUris(
        context: Context,
        uris: List<Uri>,
        title: String = "CRC Multimodal Batch Ingestion"
    ): BatchRun {
        require(uris.isNotEmpty()) { "Cannot ingest an empty list of URIs." }
        require(uris.size <= MAX_BATCH_SIZE) { "Batch size (${uris.size}) exceeds safety maximum of $MAX_BATCH_SIZE." }

        val batchId = generateBatchId()
        val contentResolver = context.contentResolver
        val items = mutableListOf<BatchSpecimenItem>()
        val seenSpecimenIds = mutableSetOf<String>()

        CrcLogger.i("[BATCH_INGEST_START] Batch: $batchId | Selected Uris: ${uris.size}")

        for (uri in uris) {
            val (fileName, fileSizeBytes) = resolveUriMetadata(contentResolver, uri)
            val startTime = System.currentTimeMillis()

            try {
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: throw IllegalArgumentException("Could not open input stream for URI: $uri")

                val bundleSha256 = computeSha256(bytes)
                val parseStart = System.currentTimeMillis()
                val bundle = AuthenticBundleLoader.loadFromBytes(bytes)
                val parseDuration = System.currentTimeMillis() - parseStart

                val specimenId = bundle.specimenId

                // Check for duplicate specimen IDs within this batch
                if (seenSpecimenIds.contains(specimenId)) {
                    CrcLogger.w("[BATCH_INGEST_DUPLICATE] Specimen '$specimenId' already present in batch $batchId. Rejecting duplicate.")
                    items.add(
                        BatchSpecimenItem(
                            batchId = batchId,
                            specimenId = specimenId,
                            sourceUri = uri.toString(),
                            fileName = fileName,
                            fileSizeBytes = bytes.size.toLong(),
                            bundleSha256 = bundleSha256,
                            visionSha256 = bundle.visionSha256,
                            textSha256 = bundle.textSha256,
                            schemaVersion = bundle.schemaVersion,
                            patchCount = bundle.patchCount,
                            status = BatchSpecimenStatus.REJECTED,
                            validationError = "Duplicate specimen identifier within batch",
                            parseTimeMs = parseDuration,
                            provenance = InputProvenance.AUTHENTIC_IMPORTED,
                            protocol = if (bundle.patchCount <= InputValidator.PROTOCOL1_MAX_PATCHES) InferenceProtocol.PROTOCOL_1_CAPPED.name else InferenceProtocol.PROTOCOL_2_UNCAPPED.name,
                            protocolReason = "Duplicate specimen identifier within batch"
                        )
                    )
                    continue
                }

                seenSpecimenIds.add(specimenId)

                // Save to private app storage
                BundleStorageManager.saveBundle(context, bundle)

                val valDuration = System.currentTimeMillis() - startTime - parseDuration
                val initialProtocol = if (bundle.patchCount <= InputValidator.PROTOCOL1_MAX_PATCHES) {
                    InferenceProtocol.PROTOCOL_1_CAPPED.name
                } else {
                    InferenceProtocol.PROTOCOL_2_UNCAPPED.name
                }
                val initialReason = if (bundle.patchCount > InputValidator.PROTOCOL1_MAX_PATCHES) {
                    "Candidate for Protocol 2 Uncapped (${bundle.patchCount} patches)"
                } else null

                items.add(
                    BatchSpecimenItem(
                        batchId = batchId,
                        specimenId = specimenId,
                        sourceUri = uri.toString(),
                        fileName = fileName,
                        fileSizeBytes = bytes.size.toLong(),
                        bundleSha256 = bundleSha256,
                        visionSha256 = bundle.visionSha256,
                        textSha256 = bundle.textSha256,
                        schemaVersion = bundle.schemaVersion,
                        patchCount = bundle.patchCount,
                        status = BatchSpecimenStatus.VALIDATED,
                        parseTimeMs = parseDuration,
                        validationTimeMs = valDuration,
                        provenance = InputProvenance.AUTHENTIC_IMPORTED,
                        protocol = initialProtocol,
                        protocolReason = initialReason
                    )
                )
                CrcLogger.i("[BATCH_INGEST_ITEM_VALID] Specimen: $specimenId | Patches: ${bundle.patchCount} | Size: ${bytes.size}B | Protocol: $initialProtocol")

            } catch (e: Exception) {
                CrcLogger.e("[BATCH_INGEST_ITEM_REJECTED] Uri: $uri | Error: ${e.message}", e)
                val fallbackId = fileName.substringBeforeLast(".").ifBlank { "UNKNOWN-${UUID.randomUUID().toString().take(4)}" }
                items.add(
                    BatchSpecimenItem(
                        batchId = batchId,
                        specimenId = fallbackId,
                        sourceUri = uri.toString(),
                        fileName = fileName,
                        fileSizeBytes = fileSizeBytes,
                        status = BatchSpecimenStatus.REJECTED,
                        validationError = e.message ?: "Failed parsing authentic bundle",
                        provenance = InputProvenance.AUTHENTIC_IMPORTED,
                        protocol = InferenceProtocol.PROTOCOL_1_CAPPED.name,
                        protocolReason = e.message ?: "Failed parsing authentic bundle"
                    )
                )
            }
        }

        val acceptedCount = items.count { it.status == BatchSpecimenStatus.VALIDATED }
        val batchStatus = if (acceptedCount > 0) BatchStatus.READY else BatchStatus.FAILED

        val batchRun = BatchRun(
            batchId = batchId,
            title = title,
            status = batchStatus,
            items = items
        )

        repository.insertBatchRun(batchRun)
        CrcLogger.i("[BATCH_INGEST_COMPLETE] Batch: $batchId | Total: ${uris.size} | Accepted: $acceptedCount | Rejected: ${items.size - acceptedCount} | Status: $batchStatus")
        return batchRun
    }

    /**
     * Ingests files directly from the filesystem (used for tests, benchmarks, or direct evaluation).
     */
    fun ingestFiles(
        files: List<File>,
        title: String = "CRC Batch Ingestion (Local Files)"
    ): BatchRun {
        require(files.isNotEmpty()) { "Cannot ingest an empty list of files." }
        require(files.size <= MAX_BATCH_SIZE) { "Batch size (${files.size}) exceeds safety maximum of $MAX_BATCH_SIZE." }

        val batchId = generateBatchId()
        val items = mutableListOf<BatchSpecimenItem>()
        val seenSpecimenIds = mutableSetOf<String>()

        CrcLogger.i("[BATCH_INGEST_FILES_START] Batch: $batchId | Files: ${files.size}")

        for (file in files) {
            val fileName = file.name
            val fileSizeBytes = file.length()
            val startTime = System.currentTimeMillis()

            try {
                val bytes = file.readBytes()
                val bundleSha256 = computeSha256(bytes)
                val parseStart = System.currentTimeMillis()
                val bundle = AuthenticBundleLoader.loadFromBytes(bytes)
                val parseDuration = System.currentTimeMillis() - parseStart

                val specimenId = bundle.specimenId

                if (seenSpecimenIds.contains(specimenId)) {
                    CrcLogger.w("[BATCH_INGEST_DUPLICATE] Specimen '$specimenId' already present in batch $batchId. Rejecting duplicate.")
                    items.add(
                        BatchSpecimenItem(
                            batchId = batchId,
                            specimenId = specimenId,
                            sourceUri = file.absolutePath,
                            fileName = fileName,
                            fileSizeBytes = fileSizeBytes,
                            bundleSha256 = bundleSha256,
                            visionSha256 = bundle.visionSha256,
                            textSha256 = bundle.textSha256,
                            schemaVersion = bundle.schemaVersion,
                            patchCount = bundle.patchCount,
                            status = BatchSpecimenStatus.REJECTED,
                            validationError = "Duplicate specimen identifier within batch",
                            parseTimeMs = parseDuration,
                            provenance = InputProvenance.AUTHENTIC_IMPORTED,
                            protocol = if (bundle.patchCount <= InputValidator.PROTOCOL1_MAX_PATCHES) InferenceProtocol.PROTOCOL_1_CAPPED.name else InferenceProtocol.PROTOCOL_2_UNCAPPED.name,
                            protocolReason = "Duplicate specimen identifier within batch"
                        )
                    )
                    continue
                }

                seenSpecimenIds.add(specimenId)

                val valDuration = System.currentTimeMillis() - startTime - parseDuration
                val initialProtocol = if (bundle.patchCount <= InputValidator.PROTOCOL1_MAX_PATCHES) {
                    InferenceProtocol.PROTOCOL_1_CAPPED.name
                } else {
                    InferenceProtocol.PROTOCOL_2_UNCAPPED.name
                }
                val initialReason = if (bundle.patchCount > InputValidator.PROTOCOL1_MAX_PATCHES) {
                    "Candidate for Protocol 2 Uncapped (${bundle.patchCount} patches)"
                } else null

                items.add(
                    BatchSpecimenItem(
                        batchId = batchId,
                        specimenId = specimenId,
                        sourceUri = file.absolutePath,
                        fileName = fileName,
                        fileSizeBytes = fileSizeBytes,
                        bundleSha256 = bundleSha256,
                        visionSha256 = bundle.visionSha256,
                        textSha256 = bundle.textSha256,
                        schemaVersion = bundle.schemaVersion,
                        patchCount = bundle.patchCount,
                        status = BatchSpecimenStatus.VALIDATED,
                        parseTimeMs = parseDuration,
                        validationTimeMs = valDuration,
                        provenance = InputProvenance.AUTHENTIC_IMPORTED,
                        protocol = initialProtocol,
                        protocolReason = initialReason
                    )
                )
                CrcLogger.i("[BATCH_INGEST_FILE_VALID] Specimen: $specimenId | Patches: ${bundle.patchCount} | Size: ${fileSizeBytes}B | Protocol: $initialProtocol")

            } catch (e: Exception) {
                CrcLogger.e("[BATCH_INGEST_FILE_ERROR] Failed to ingest $fileName: ${e.message}", e)
                val parseDuration = System.currentTimeMillis() - startTime
                items.add(
                    BatchSpecimenItem(
                        batchId = batchId,
                        specimenId = fileName.removeSuffix(".crcbundle"),
                        sourceUri = file.absolutePath,
                        fileName = fileName,
                        fileSizeBytes = fileSizeBytes,
                        bundleSha256 = "",
                        visionSha256 = "",
                        textSha256 = "",
                        schemaVersion = 0,
                        patchCount = 0,
                        status = BatchSpecimenStatus.REJECTED,
                        validationError = e.message ?: "Unknown validation error",
                        parseTimeMs = parseDuration,
                        provenance = InputProvenance.AUTHENTIC_IMPORTED,
                        protocol = InferenceProtocol.PROTOCOL_1_CAPPED.name,
                        protocolReason = e.message ?: "Unknown validation error"
                    )
                )
            }
        }

        val acceptedCount = items.count { it.status == BatchSpecimenStatus.VALIDATED }
        val batchStatus = if (acceptedCount > 0) BatchStatus.READY else BatchStatus.FAILED

        val batchRun = BatchRun(
            batchId = batchId,
            title = title,
            status = batchStatus,
            items = items
        )

        repository.insertBatchRun(batchRun)
        CrcLogger.i("[BATCH_INGEST_FILES_COMPLETE] Batch: $batchId | Total: ${files.size} | Accepted: $acceptedCount | Rejected: ${items.size - acceptedCount} | Status: $batchStatus")
        return batchRun
    }


    /**
     * Executes controlled sequential batch inference on validated items.
     * Supports Protocol 1 (capped at 1,000 patches) with automatic or configurable
     * routing to Protocol 2 (uncapped) for specimens exceeding the Protocol 1 limit.
     */
    fun executeBatchInference(
        context: Context,
        batchRun: BatchRun,
        classifier: CrcMultimodalClassifier,
        onProgress: (completed: Int, total: Int, currentSpecimen: String) -> Unit = { _, _, _ -> },
        isCancelled: () -> Boolean = { false },
        baseProtocol: InferenceProtocol = InferenceProtocol.PROTOCOL_1_CAPPED,
        allowProtocol2Fallback: Boolean = true
    ): BatchRun {
        var currentBatch = batchRun.copy(
            status = BatchStatus.PROCESSING,
            updatedAt = System.currentTimeMillis()
        )
        repository.updateBatchRun(currentBatch)

        val updatedItems = currentBatch.items.toMutableList()
        val validItems = updatedItems.filter { it.status == BatchSpecimenStatus.VALIDATED }
        val totalToProcess = validItems.size
        var completedCount = 0
        val batchStartTime = System.currentTimeMillis()

        CrcLogger.i("[BATCH_INFERENCE_START] Batch: ${batchRun.batchId} | ItemsToProcess: $totalToProcess | BaseProtocol: ${baseProtocol.name} | AllowP2Fallback: $allowProtocol2Fallback")

        for (i in updatedItems.indices) {
            val item = updatedItems[i]
            if (item.status != BatchSpecimenStatus.VALIDATED) {
                continue
            }

            // Check cancellation before each specimen
            if (isCancelled()) {
                CrcLogger.w("[BATCH_INFERENCE_CANCELLED] Batch: ${batchRun.batchId} cancelled by user at specimen ${item.specimenId}.")
                // Mark current and subsequent validated items as CANCELLED
                for (j in i until updatedItems.size) {
                    if (updatedItems[j].status == BatchSpecimenStatus.VALIDATED) {
                        updatedItems[j] = updatedItems[j].copy(
                            status = BatchSpecimenStatus.CANCELLED,
                            errorDetails = "Cancelled by user before execution"
                        )
                        repository.updateBatchItem(updatedItems[j])
                    }
                }
                break
            }

            onProgress(completedCount, totalToProcess, item.specimenId)

            var targetProtocol = baseProtocol
            var protocolReason: String? = null

            try {
                val sourceFile = if (item.sourceUri.isNotBlank()) File(item.sourceUri) else null
                val bundle = if (sourceFile != null && sourceFile.exists()) {
                    AuthenticBundleLoader.loadFromFile(sourceFile)
                } else {
                    BundleStorageManager.loadBundle(context, item.specimenId)
                } ?: throw IllegalStateException("Cached bundle for specimen '${item.specimenId}' not found in storage.")

                // Evaluate patch count against protocol boundaries
                if (bundle.patchCount <= InputValidator.PROTOCOL1_MAX_PATCHES) {
                    targetProtocol = baseProtocol
                    protocolReason = null
                } else if (allowProtocol2Fallback) {
                    targetProtocol = InferenceProtocol.PROTOCOL_2_UNCAPPED
                    protocolReason = "Exceeded Protocol 1 cap (${bundle.patchCount} > ${InputValidator.PROTOCOL1_MAX_PATCHES} patches); routed to Protocol 2 Uncapped"
                    CrcLogger.i("[BATCH_PROTOCOL_ROUTING] Specimen '${item.specimenId}' with ${bundle.patchCount} patches automatically routed from Protocol 1 to Protocol 2 Uncapped.")
                } else {
                    protocolReason = "Exceeded Protocol 1 cap (${bundle.patchCount} > ${InputValidator.PROTOCOL1_MAX_PATCHES} patches)"
                    CrcLogger.w("[BATCH_INFERENCE_POLICY_REJECT] Specimen '${item.specimenId}' with ${bundle.patchCount} patches rejected: Protocol 1 cap exceeded and Protocol 2 fallback disabled.")
                    val rejectedItem = item.copy(
                        status = BatchSpecimenStatus.REJECTED,
                        validationError = protocolReason,
                        protocol = InferenceProtocol.PROTOCOL_1_CAPPED.name,
                        protocolReason = protocolReason,
                        completedTimestamp = System.currentTimeMillis()
                    )
                    updatedItems[i] = rejectedItem
                    repository.updateBatchItem(rejectedItem)
                    continue
                }

                val inferStartTime = System.currentTimeMillis()
                val result = classifier.predictAuthentic(
                    visionData = bundle.visionData,
                    numPatches = bundle.patchCount,
                    textData = bundle.textData,
                    caseId = bundle.specimenId,
                    provenance = InputProvenance.AUTHENTIC_IMPORTED,
                    protocol = targetProtocol
                )
                val inferLatency = System.currentTimeMillis() - inferStartTime

                val completedItem = item.copy(
                    status = BatchSpecimenStatus.COMPLETED,
                    logit = result.logit,
                    probability = result.probability,
                    predictedClass = result.predictedClass,
                    predictedLabel = result.predictedLabel,
                    threshold = result.threshold,
                    inferenceLatencyMs = inferLatency,
                    provenance = result.provenance,
                    completedTimestamp = System.currentTimeMillis(),
                    protocol = targetProtocol.name,
                    protocolReason = protocolReason
                )
                updatedItems[i] = completedItem
                repository.updateBatchItem(completedItem)

                // Persist to shared Cases registry
                persistCompletedSpecimenToCases(context, completedItem, workstationRepository)

                completedCount++

                CrcLogger.i(
                    "[BATCH_INFERENCE_ITEM_SUCCESS] Specimen: ${item.specimenId} | " +
                    "Protocol: ${targetProtocol.name} | " +
                    "Prob: ${String.format(Locale.US, "%.5f", result.probability)} | " +
                    "Class: ${result.predictedClass} (${result.predictedLabel}) | " +
                    "Latency: ${inferLatency}ms"
                )

            } catch (e: Exception) {
                CrcLogger.e("[BATCH_INFERENCE_ITEM_FAILED] Specimen: ${item.specimenId} | Protocol: ${targetProtocol.name} | Error: ${e.message}", e)
                val failedItem = item.copy(
                    status = BatchSpecimenStatus.FAILED,
                    errorDetails = e.message ?: "Unknown inference error",
                    completedTimestamp = System.currentTimeMillis(),
                    protocol = targetProtocol.name,
                    protocolReason = protocolReason
                )
                updatedItems[i] = failedItem
                repository.updateBatchItem(failedItem)
            }
        }

        val totalDuration = System.currentTimeMillis() - batchStartTime
        val successfulCount = updatedItems.count { it.status == BatchSpecimenStatus.COMPLETED }
        val failedCount = updatedItems.count { it.status == BatchSpecimenStatus.FAILED }
        val cancelledCount = updatedItems.count { it.status == BatchSpecimenStatus.CANCELLED }

        val terminalStatus = when {
            isCancelled() -> BatchStatus.CANCELLED
            cancelledCount > 0 -> BatchStatus.PARTIALLY_COMPLETED
            failedCount > 0 && successfulCount > 0 -> BatchStatus.PARTIALLY_COMPLETED
            failedCount > 0 && successfulCount == 0 -> BatchStatus.FAILED
            else -> BatchStatus.COMPLETED
        }

        val finalizedBatch = currentBatch.copy(
            status = terminalStatus,
            totalDurationMs = totalDuration,
            updatedAt = System.currentTimeMillis(),
            items = updatedItems
        )

        repository.updateBatchRun(finalizedBatch)
        onProgress(completedCount, totalToProcess, "Complete")
        CrcLogger.i("[BATCH_INFERENCE_COMPLETE] Batch: ${batchRun.batchId} | Status: $terminalStatus | Success: $successfulCount | Failed: $failedCount | Duration: ${totalDuration}ms")

        return finalizedBatch
    }

    private fun resolveUriMetadata(contentResolver: ContentResolver, uri: Uri): Pair<String, Long> {
        var name = uri.lastPathSegment ?: "unknown.crcbundle"
        var size = 0L

        if (uri.scheme == ContentResolver.SCHEME_CONTENT) {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                val sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE)
                if (cursor.moveToFirst()) {
                    if (nameIndex != -1) name = cursor.getString(nameIndex)
                    if (sizeIndex != -1) size = cursor.getLong(sizeIndex)
                }
            }
        } else if (uri.scheme == "file") {
            val file = File(uri.path ?: "")
            if (file.exists()) {
                name = file.name
                size = file.length()
            }
        }
        return Pair(name, size)
    }

    private fun computeSha256(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256")
        return digest.digest(bytes).joinToString("") { "%02x".format(it) }
    }
}
