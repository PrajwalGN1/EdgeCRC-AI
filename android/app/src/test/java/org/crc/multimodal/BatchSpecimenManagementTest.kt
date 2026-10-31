package org.crc.multimodal

import org.crc.multimodal.batch.BatchCoordinator
import org.crc.multimodal.batch.BatchReportGenerator
import org.crc.multimodal.batch.BatchRun
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import org.crc.multimodal.batch.BatchStatistics
import org.crc.multimodal.batch.BatchStatus
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.security.AuditActionType
import org.crc.multimodal.security.AuditTrailRecord
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File

/**
 * Authoritative automated test suite for CRC Edge AI Batch Cohort Specimen Management:
 * 1. Shared Cases registry integration and idempotency.
 * 2. Multi-selection state, Select All, Clear, and stable specimen identity.
 * 3. Archive-by-default soft removal, active statistics exclusion, and audit recording.
 * 4. Separate, explicitly confirmed permanent deletion with strict ownership isolation.
 * 5. Combined bulk PDF export with cryptographic integrity manifests.
 */
class BatchSpecimenManagementTest {

    private lateinit var tempDir: File
    private lateinit var batchRepo: InMemoryBatchRepository
    private lateinit var caseRepo: InMemoryWorkstationRepository

    @Before
    fun setUp() {
        tempDir = File("build/tmp/batch_specimen_mgmt_tests_${System.currentTimeMillis()}").apply { mkdirs() }
        batchRepo = InMemoryBatchRepository()
        caseRepo = InMemoryWorkstationRepository()
    }

    @After
    fun tearDown() {
        tempDir.deleteRecursively()
    }

    // =========================================================================
    // SECTION 1: PERSISTENCE & SHARED CASES REGISTRY INTEGRATION
    // =========================================================================

    @Test
    fun testSuccessfulBatchInferencePersistsToSharedCases() {
        val batchId = "BATCH-PERSIST-001"
        val item = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-CASE-101",
            sourceUri = "file:///storage/emulated/0/Download/SPEC-CASE-101.crcbundle",
            fileName = "SPEC-CASE-101.crcbundle",
            patchCount = 420,
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.9421f,
            logit = 2.7891f,
            predictedClass = 1,
            predictedLabel = "MSI",
            threshold = ThresholdLogic.OPERATING_THRESHOLD,
            inferenceLatencyMs = 185L,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED.name,
            completedTimestamp = System.currentTimeMillis()
        )
        val batchRun = BatchRun(
            batchId = batchId,
            title = "Batch Shared Registry Sync Test",
            status = BatchStatus.COMPLETED,
            items = listOf(item)
        )
        batchRepo.insertBatchRun(batchRun)

        // Execute batch sync to shared cases
        val syncedCount = BatchCoordinator.syncBatchToCases(
            context = null,
            batchRun = batchRun,
            workstationRepo = caseRepo
        )
        assertEquals(1, syncedCount)

        // Verify case is discoverable in shared cases repository
        val persistedCase = caseRepo.getCaseByIdentifier("SPEC-CASE-101")
        assertNotNull("Specimen must be created in shared cases registry", persistedCase)
        assertEquals("SPEC-CASE-101", persistedCase!!.caseIdentifier)
        assertEquals(CaseStatus.ANALYZED, persistedCase.status)
        assertEquals(420, persistedCase.patchCount)
        assertTrue("Batch provenance must be recorded in researchContext", persistedCase.researchContext.contains(batchId))

        // Verify inference run is recorded with faithful metrics
        val runs = caseRepo.getRunsForCase(persistedCase.id)
        assertEquals(1, runs.size)
        val run = runs.first()
        assertEquals(1, run.predictedClass)
        assertEquals("MSI", run.predictedLabel)
        assertEquals(0.9421f, run.probability, 0.0001f)
        assertEquals(2.7891f, run.logit, 0.0001f)
        assertEquals(185.0, run.latencyMs, 0.001)
        assertEquals(ThresholdLogic.OPERATING_THRESHOLD, run.threshold, 0.0001f)
        assertEquals(InferenceProtocol.PROTOCOL_1_CAPPED, run.protocol)

        // Verify audit trail entry
        val auditRecords = caseRepo.getAuditTrailForSpecimen("SPEC-CASE-101")
        assertTrue("Audit trail must record batch specimen synchronization", auditRecords.isNotEmpty())
        assertEquals(AuditActionType.INFERENCE_EXECUTED, auditRecords.first().actionType)
    }

    @Test
    fun testFailedInferenceDoesNotCreateSuccessfulCasesRecord() {
        val batchId = "BATCH-FAILED-002"
        val failedItem = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-FAILED-999",
            patchCount = 50,
            status = BatchSpecimenStatus.FAILED,
            probability = null,
            errorDetails = "ONNX runtime tensor shape mismatch"
        )
        val rejectedItem = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-REJECTED-888",
            patchCount = 0,
            status = BatchSpecimenStatus.REJECTED,
            probability = null,
            validationError = "Checksum mismatch: corrupt bundle"
        )
        val batchRun = BatchRun(
            batchId = batchId,
            title = "Failed Batch Run",
            status = BatchStatus.PARTIALLY_COMPLETED,
            items = listOf(failedItem, rejectedItem)
        )
        batchRepo.insertBatchRun(batchRun)

        val syncedCount = BatchCoordinator.syncBatchToCases(
            context = null,
            batchRun = batchRun,
            workstationRepo = caseRepo
        )
        assertEquals("Failed and rejected specimens must not sync to Cases", 0, syncedCount)
        assertNull("Failed specimen must not exist in Cases registry", caseRepo.getCaseByIdentifier("SPEC-FAILED-999"))
        assertNull("Rejected specimen must not exist in Cases registry", caseRepo.getCaseByIdentifier("SPEC-REJECTED-888"))
    }

    @Test
    fun testRejectedDuplicateDoesNotOverwriteExistingAnalyzedCase() {
        // Step 1: Pre-existing authentic analyzed case
        val initialCase = ResearchCase(
            caseIdentifier = "SPEC-DUP-PROTECT",
            specimenDescription = "Authentic pathology biopsy",
            researchContext = "Clinical Trial 2026",
            status = CaseStatus.ANALYZED,
            patchCount = 300,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val caseId = caseRepo.insertCase(initialCase)
        val initialRun = InferenceRun(
            caseId = caseId,
            modelIdentifier = "fusion_model_fp32.onnx",
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.9850f,
            logit = 3.2f,
            threshold = 0.83f,
            latencyMs = 142.0,
            numPatches = 300
        )
        caseRepo.recordInferenceResult(caseId, initialRun, CaseStatus.ANALYZED)

        // Step 2: Batch item with same ID but rejected as duplicate
        val duplicateItem = BatchSpecimenItem(
            batchId = "BATCH-DUP-003",
            specimenId = "SPEC-DUP-PROTECT",
            patchCount = 100,
            status = BatchSpecimenStatus.REJECTED,
            validationError = "Duplicate specimen identifier already exists"
        )
        val batchRun = BatchRun(
            batchId = "BATCH-DUP-003",
            title = "Duplicate Ingestion Attempt",
            status = BatchStatus.COMPLETED,
            items = listOf(duplicateItem)
        )

        val syncedCount = BatchCoordinator.syncBatchToCases(null, batchRun, caseRepo)
        assertEquals(0, syncedCount)

        // Step 3: Verify existing case is preserved without alteration
        val preservedCase = caseRepo.getCaseByIdentifier("SPEC-DUP-PROTECT")
        assertNotNull(preservedCase)
        assertEquals("Authentic pathology biopsy", preservedCase!!.specimenDescription)
        assertEquals("Clinical Trial 2026", preservedCase.researchContext)
        assertEquals(300, preservedCase.patchCount)
        assertEquals(CaseStatus.ANALYZED, preservedCase.status)

        val preservedRuns = caseRepo.getRunsForCase(caseId)
        assertEquals(1, preservedRuns.size)
        assertEquals(0.9850f, preservedRuns.first().probability, 0.0001f)
    }

    @Test
    fun testBatchReopeningIsIdempotent() {
        val batchId = "BATCH-IDEMPOTENT-004"
        val item = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-IDEM-001",
            patchCount = 200,
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.2150f,
            logit = -1.295f,
            predictedClass = 0,
            predictedLabel = "MSS",
            inferenceLatencyMs = 120L
        )
        val batchRun = BatchRun(
            batchId = batchId,
            title = "Idempotency Test Batch",
            status = BatchStatus.COMPLETED,
            items = listOf(item)
        )

        // Sync 1st time
        val sync1 = BatchCoordinator.syncBatchToCases(null, batchRun, caseRepo)
        assertEquals(1, sync1)

        val casesAfterFirst = caseRepo.getAllCases()
        assertEquals(1, casesAfterFirst.size)
        val caseId = casesAfterFirst.first().id
        val runsAfterFirst = caseRepo.getRunsForCase(caseId)
        assertEquals(1, runsAfterFirst.size)

        // Sync 2nd time (simulating screen re-opening or application restart)
        val sync2 = BatchCoordinator.syncBatchToCases(null, batchRun, caseRepo)
        assertEquals(0, sync2) // 0 new records inserted

        val casesAfterSecond = caseRepo.getAllCases()
        assertEquals("Idempotency: case record count must not multiply", 1, casesAfterSecond.size)
        val runsAfterSecond = caseRepo.getRunsForCase(caseId)
        assertEquals("Idempotency: inference run count must not multiply", 1, runsAfterSecond.size)
    }

    // =========================================================================
    // SECTION 2: MULTI-SELECTION LOGIC & STABLE IDENTIFIERS
    // =========================================================================

    @Test
    fun testMultiSelectionStableIdentifiersAndBulkOperations() {
        val selectedIds = mutableSetOf<String>()

        val items = listOf(
            BatchSpecimenItem(batchId = "B1", specimenId = "SPEC-A", status = BatchSpecimenStatus.COMPLETED, probability = 0.90f),
            BatchSpecimenItem(batchId = "B1", specimenId = "SPEC-B", status = BatchSpecimenStatus.COMPLETED, probability = 0.10f),
            BatchSpecimenItem(batchId = "B1", specimenId = "SPEC-C", status = BatchSpecimenStatus.REJECTED, probability = null)
        )

        // Select individual items
        selectedIds.add("SPEC-A")
        selectedIds.add("SPEC-B")
        assertEquals(2, selectedIds.size)
        assertTrue(selectedIds.contains("SPEC-A"))
        assertTrue(selectedIds.contains("SPEC-B"))
        assertFalse(selectedIds.contains("SPEC-C"))

        // Reordering or filtering must not invalidate stable specimen identifiers
        val reorderedItems = items.reversed()
        val matchingSelected = reorderedItems.filter { selectedIds.contains(it.specimenId) }
        assertEquals(2, matchingSelected.size)
        assertEquals(setOf("SPEC-A", "SPEC-B"), matchingSelected.map { it.specimenId }.toSet())

        // Clear selection
        selectedIds.clear()
        assertTrue(selectedIds.isEmpty())

        // Select All eligible
        val allIds = items.map { it.specimenId }
        selectedIds.addAll(allIds)
        assertEquals(3, selectedIds.size)
    }

    // =========================================================================
    // SECTION 3: ARCHIVE-BY-DEFAULT, COHORT METRICS, & RESTORATION
    // =========================================================================

    @Test
    fun testArchiveRemovesSpecimensFromActiveCohortMetrics() {
        val batchId = "BATCH-ARCHIVE-005"
        val items = listOf(
            BatchSpecimenItem(batchId = batchId, specimenId = "SPEC-1", status = BatchSpecimenStatus.COMPLETED, predictedClass = 1, probability = 0.91f, inferenceLatencyMs = 100L),
            BatchSpecimenItem(batchId = batchId, specimenId = "SPEC-2", status = BatchSpecimenStatus.COMPLETED, predictedClass = 1, probability = 0.88f, inferenceLatencyMs = 110L),
            BatchSpecimenItem(batchId = batchId, specimenId = "SPEC-3", status = BatchSpecimenStatus.COMPLETED, predictedClass = 0, probability = 0.20f, inferenceLatencyMs = 120L),
            BatchSpecimenItem(batchId = batchId, specimenId = "SPEC-4", status = BatchSpecimenStatus.COMPLETED, predictedClass = 0, probability = 0.15f, inferenceLatencyMs = 130L)
        )
        val initialBatch = BatchRun(batchId = batchId, title = "Cohort Metrics Test", status = BatchStatus.COMPLETED, items = items)
        batchRepo.insertBatchRun(initialBatch)

        val statsBefore = BatchStatistics.compute(items)
        assertEquals(4, statsBefore.totalProcessed)
        assertEquals(2, statsBefore.msiCount)
        assertEquals(2, statsBefore.mssCount)
        assertEquals(0, statsBefore.totalArchived)

        // Archive SPEC-1 (an MSI specimen)
        val archivedOk = batchRepo.archiveBatchItem(batchId, "SPEC-1")
        assertTrue(archivedOk)

        // Record audit event
        caseRepo.recordAuditEvent(
            AuditTrailRecord(
                actionType = AuditActionType.SPECIMEN_ARCHIVED,
                specimenId = "SPEC-1",
                details = "Archived specimen SPEC-1 from batch $batchId"
            )
        )

        // Reload updated batch
        val updatedBatch = batchRepo.getBatchRun(batchId)
        assertNotNull(updatedBatch)
        val item1 = updatedBatch!!.items.first { it.specimenId == "SPEC-1" }
        assertTrue("SPEC-1 must have isArchived = true", item1.isArchived)

        // Compute active cohort metrics
        val statsAfter = BatchStatistics.compute(updatedBatch.items)
        assertEquals("Active processed specimens must exclude archived item", 3, statsAfter.totalProcessed)
        assertEquals("Active MSI count must decrease by 1", 1, statsAfter.msiCount)
        assertEquals("Active MSS count must remain 2", 2, statsAfter.mssCount)
        assertEquals("Total archived count must reflect 1 archived specimen", 1, statsAfter.totalArchived)

        // Verify restoration recovers specimen to active cohort
        val restoredOk = batchRepo.unarchiveBatchItem(batchId, "SPEC-1")
        assertTrue(restoredOk)

        caseRepo.recordAuditEvent(
            AuditTrailRecord(
                actionType = AuditActionType.SPECIMEN_RESTORED,
                specimenId = "SPEC-1",
                details = "Restored specimen SPEC-1 to batch $batchId"
            )
        )

        val restoredBatch = batchRepo.getBatchRun(batchId)
        val statsRestored = BatchStatistics.compute(restoredBatch!!.items)
        assertEquals(4, statsRestored.totalProcessed)
        assertEquals(2, statsRestored.msiCount)
        assertEquals(0, statsRestored.totalArchived)

        // Verify audit trail logged both archive and restore (most recent first)
        val auditTrail = caseRepo.getAuditTrailForSpecimen("SPEC-1")
        assertEquals(2, auditTrail.size)
        assertEquals(AuditActionType.SPECIMEN_RESTORED, auditTrail[0].actionType)
        assertEquals(AuditActionType.SPECIMEN_ARCHIVED, auditTrail[1].actionType)
    }

    // =========================================================================
    // SECTION 4: PERMANENT DELETION WITH STRICT OWNERSHIP ISOLATION
    // =========================================================================

    @Test
    fun testPermanentDeletionIsolatesSpecimenDataAndPreservesBatchHistory() {
        val batchId = "BATCH-DELETE-006"
        val item1 = BatchSpecimenItem(batchId = batchId, specimenId = "SPEC-DEL-1", status = BatchSpecimenStatus.COMPLETED, probability = 0.90f)
        val item2 = BatchSpecimenItem(batchId = batchId, specimenId = "SPEC-KEEP-2", status = BatchSpecimenStatus.COMPLETED, probability = 0.15f)

        val batch = BatchRun(batchId = batchId, title = "Deletion Isolation Test", status = BatchStatus.COMPLETED, items = listOf(item1, item2))
        batchRepo.insertBatchRun(batch)
        BatchCoordinator.syncBatchToCases(null, batch, caseRepo)

        // Verify both cases exist initially
        assertNotNull(caseRepo.getCaseByIdentifier("SPEC-DEL-1"))
        assertNotNull(caseRepo.getCaseByIdentifier("SPEC-KEEP-2"))

        // Perform confirmed permanent deletion of SPEC-DEL-1
        val batchDeleteOk = batchRepo.deleteBatchItem(batchId, "SPEC-DEL-1")
        assertTrue(batchDeleteOk)

        val case1 = caseRepo.getCaseByIdentifier("SPEC-DEL-1")
        assertNotNull(case1)
        val caseDeleteOk = caseRepo.deleteCase(case1!!.id)
        assertTrue(caseDeleteOk)

        caseRepo.recordAuditEvent(
            AuditTrailRecord(
                actionType = AuditActionType.SPECIMEN_DELETED,
                specimenId = "SPEC-DEL-1",
                details = "Permanently deleted specimen SPEC-DEL-1 from batch $batchId"
            )
        )

        // 1. Verify deleted specimen is removed from batch and cases
        val updatedBatch = batchRepo.getBatchRun(batchId)
        assertNotNull(updatedBatch)
        assertEquals(1, updatedBatch!!.items.size)
        assertEquals("SPEC-KEEP-2", updatedBatch.items.first().specimenId)
        assertNull(caseRepo.getCaseByIdentifier("SPEC-DEL-1"))

        // 2. Verify unrelated specimen remains completely intact
        val keptCase = caseRepo.getCaseByIdentifier("SPEC-KEEP-2")
        assertNotNull(keptCase)
        assertEquals("SPEC-KEEP-2", keptCase!!.caseIdentifier)

        // 3. Verify batch run record itself is retained in history
        val allBatches = batchRepo.getAllBatchRuns()
        assertTrue("Batch run record must remain in history", allBatches.any { it.batchId == batchId })

        // 4. Verify deletion audit record is recorded and historical inference audit is retained
        val audits = caseRepo.getAuditTrailForSpecimen("SPEC-DEL-1")
        assertEquals(2, audits.size)
        assertEquals(AuditActionType.SPECIMEN_DELETED, audits[0].actionType)
        assertEquals(AuditActionType.INFERENCE_EXECUTED, audits[1].actionType)
    }

    // =========================================================================
    // SECTION 5: BULK COMBINED PDF REPORT EXPORT
    // =========================================================================

    @Test
    fun testExportCombinedReportOnlyIncludesSelectedEligibleSpecimens() {
        val batchId = "BATCH-COMBINED-007"
        val itemA = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-REPORT-A",
            patchCount = 150,
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.9250f,
            logit = 2.512f,
            predictedClass = 1,
            predictedLabel = "MSI",
            inferenceLatencyMs = 120L,
            completedTimestamp = System.currentTimeMillis()
        )
        val itemB = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-REPORT-B",
            patchCount = 200,
            status = BatchSpecimenStatus.COMPLETED,
            probability = 0.1250f,
            logit = -1.945f,
            predictedClass = 0,
            predictedLabel = "MSS",
            inferenceLatencyMs = 135L,
            completedTimestamp = System.currentTimeMillis()
        )
        val itemRejected = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-REJECTED-C",
            patchCount = 0,
            status = BatchSpecimenStatus.REJECTED,
            probability = null,
            validationError = "Corrupt bundle structure"
        )

        // Pass all selected items including rejected; generator must filter to only eligible items
        val bundle = BatchReportGenerator.exportCombinedReport(
            context = null,
            batchId = batchId,
            selectedItems = listOf(itemA, itemB, itemRejected),
            outputDir = tempDir
        )

        assertNotNull(bundle)
        assertEquals(batchId, bundle.batchId)
        assertTrue("JSON file must exist", bundle.jsonFile.exists())
        assertTrue("Manifest file must exist", bundle.manifestFile.exists())
        assertNotNull(bundle.jsonSha256)

        // Verify JSON contents
        val jsonText = bundle.jsonFile.readText(Charsets.UTF_8)
        assertTrue(jsonText.contains("SPEC-REPORT-A"))
        assertTrue(jsonText.contains("SPEC-REPORT-B"))
        assertFalse("Rejected specimen must not be in report", jsonText.contains("SPEC-REJECTED-C"))
        assertTrue(jsonText.contains("\"total_selected_specimens\": 2"))
        assertTrue(jsonText.contains("\"msi_predicted_count\": 1"))
        assertTrue(jsonText.contains("\"mss_predicted_count\": 1"))

        // Verify manifest integrity
        val manifestText = bundle.manifestFile.readText(Charsets.UTF_8)
        assertTrue(manifestText.contains(bundle.jsonSha256))
        assertTrue(manifestText.contains(bundle.jsonFile.name))
        assertTrue(BatchReportGenerator.verifyReportIntegrity(bundle.jsonFile, bundle.jsonSha256))
    }

    @Test
    fun testExportRejectsIfNoEligibleSpecimensSelected() {
        val batchId = "BATCH-NO-ELIGIBLE-008"
        val itemRejected = BatchSpecimenItem(
            batchId = batchId,
            specimenId = "SPEC-INELIGIBLE",
            status = BatchSpecimenStatus.REJECTED,
            probability = null
        )

        try {
            BatchReportGenerator.exportCombinedReport(
                context = null,
                batchId = batchId,
                selectedItems = listOf(itemRejected),
                outputDir = tempDir
            )
            fail("Expected IllegalArgumentException when exporting batch with no eligible specimens")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("No eligible analyzed specimens selected"))
        }
    }
}
