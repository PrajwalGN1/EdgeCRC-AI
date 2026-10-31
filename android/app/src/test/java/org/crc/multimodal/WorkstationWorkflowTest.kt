package org.crc.multimodal

import org.crc.multimodal.data.CaseSortOption
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.data.ReviewNote
import org.crc.multimodal.data.ReviewNoteCategory
import org.crc.multimodal.data.SpecimenInput
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.InputValidator
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.report.ReportGenerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test

/**
 * Comprehensive workflow and data integrity unit tests for the CRC Multimodal Pathology Research Workstation.
 */
class WorkstationWorkflowTest {

    private lateinit var repository: InMemoryWorkstationRepository

    @Before
    fun setUp() {
        repository = InMemoryWorkstationRepository()
    }

    @Test
    fun testCaseCreationAndValidation() {
        // Valid case creation
        val validCase = ResearchCase(
            caseIdentifier = "CRC-TEST-001",
            specimenDescription = "Sigmoid colon resection",
            researchContext = "Primary adenocarcinoma staging",
            status = CaseStatus.DRAFT,
            patchCount = 50,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val caseId = repository.insertCase(validCase)
        assertTrue(caseId > 0)

        val retrieved = repository.getCaseById(caseId)
        assertNotNull(retrieved)
        assertEquals("CRC-TEST-001", retrieved?.caseIdentifier)
        assertEquals(50, retrieved?.patchCount)
        assertEquals(CaseStatus.DRAFT, retrieved?.status)

        // Blank identifier rejection
        try {
            ResearchCase(caseIdentifier = "   ")
            fail("Expected exception for blank case identifier")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("blank"))
        }

        // Negative patch count rejection
        try {
            ResearchCase(caseIdentifier = "CRC-TEST-002", patchCount = -5)
            fail("Expected exception for non-positive patch count")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("positive"))
        }
    }

    @Test
    fun testDuplicateCaseIdentifierPrevention() {
        val case1 = ResearchCase(caseIdentifier = "CRC-UNIQUE-01", specimenDescription = "Biopsy A")
        repository.insertCase(case1)

        // Attempt duplicate insertion
        val duplicateCase = ResearchCase(caseIdentifier = "crc-unique-01", specimenDescription = "Biopsy B")
        try {
            repository.insertCase(duplicateCase)
            fail("Expected exception when inserting duplicate case identifier")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("already exists"))
        }

        // Attempt updating another case to an existing identifier
        val case2 = ResearchCase(caseIdentifier = "CRC-UNIQUE-02", specimenDescription = "Biopsy C")
        val case2Id = repository.insertCase(case2)

        try {
            repository.updateCase(case2.copy(id = case2Id, caseIdentifier = "CRC-UNIQUE-01"))
            fail("Expected exception when updating to an existing identifier")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("already exists"))
        }
    }

    @Test
    fun testDraftPersistenceAndCaseReopening() {
        val initialDraft = ResearchCase(
            caseIdentifier = "CRC-DRAFT-001",
            specimenDescription = "Cecum biopsy",
            researchContext = "Initial intake notes",
            status = CaseStatus.DRAFT,
            patchCount = 10
        )
        val id = repository.insertCase(initialDraft)

        // Simulate user reopening and modifying draft
        val reopened = repository.getCaseById(id)
        assertNotNull(reopened)
        assertEquals(CaseStatus.DRAFT, reopened?.status)

        val updatedDraft = reopened!!.copy(
            specimenDescription = "Cecum biopsy FFPE with lymphocytic infiltrate",
            researchContext = "Patient presenting with right lower quadrant mass",
            patchCount = 100,
            status = CaseStatus.READY_FOR_ANALYSIS
        )
        assertTrue(repository.updateCase(updatedDraft))

        val retrievedAfterUpdate = repository.getCaseById(id)
        assertNotNull(retrievedAfterUpdate)
        assertEquals("Cecum biopsy FFPE with lymphocytic infiltrate", retrievedAfterUpdate?.specimenDescription)
        assertEquals(CaseStatus.READY_FOR_ANALYSIS, retrievedAfterUpdate?.status)
        assertEquals(100, retrievedAfterUpdate?.patchCount)
    }

    @Test
    fun testFilterAndSearchWorkflow() {
        repository.insertCase(ResearchCase(caseIdentifier = "CASE-A-DRAFT", specimenDescription = "Colon tissue", status = CaseStatus.DRAFT))
        repository.insertCase(ResearchCase(caseIdentifier = "CASE-B-READY", specimenDescription = "Biopsy FFPE", status = CaseStatus.READY_FOR_ANALYSIS))
        repository.insertCase(ResearchCase(caseIdentifier = "CASE-C-ANALYZED", specimenDescription = "Resection specimen", status = CaseStatus.ANALYZED))
        repository.insertCase(ResearchCase(caseIdentifier = "CASE-D-REVIEW", specimenDescription = "Colon polyp", status = CaseStatus.UNDER_REVIEW))
        repository.insertCase(ResearchCase(caseIdentifier = "CASE-E-FINAL", specimenDescription = "Mucinous carcinoma", status = CaseStatus.REVIEWED))

        // Status filters
        assertEquals(1, repository.getCasesFiltered(status = CaseStatus.DRAFT).size)
        assertEquals(1, repository.getCasesFiltered(status = CaseStatus.READY_FOR_ANALYSIS).size)
        assertEquals(1, repository.getCasesFiltered(status = CaseStatus.ANALYZED).size)
        assertEquals(1, repository.getCasesFiltered(status = CaseStatus.UNDER_REVIEW).size)
        assertEquals(1, repository.getCasesFiltered(status = CaseStatus.REVIEWED).size)
        assertEquals(5, repository.getCasesFiltered(status = null).size)

        // Text search
        val searchColon = repository.getCasesFiltered(searchQuery = "Colon")
        assertEquals(2, searchColon.size)

        val searchMucinous = repository.getCasesFiltered(searchQuery = "mucinous")
        assertEquals(1, searchMucinous.size)
        assertEquals("CASE-E-FINAL", searchMucinous[0].caseIdentifier)

        // Sort by case identifier
        val sortedById = repository.getCasesFiltered(sortBy = CaseSortOption.CASE_IDENTIFIER)
        assertEquals("CASE-A-DRAFT", sortedById.first().caseIdentifier)
        assertEquals("CASE-E-FINAL", sortedById.last().caseIdentifier)
    }

    @Test
    fun testDashboardMetricsCalculation() {
        assertEquals(0, repository.getDashboardCounts().totalCases)

        repository.insertCase(ResearchCase(caseIdentifier = "C1", status = CaseStatus.DRAFT))
        repository.insertCase(ResearchCase(caseIdentifier = "C2", status = CaseStatus.DRAFT))
        repository.insertCase(ResearchCase(caseIdentifier = "C3", status = CaseStatus.READY_FOR_ANALYSIS))
        repository.insertCase(ResearchCase(caseIdentifier = "C4", status = CaseStatus.ANALYZED))
        repository.insertCase(ResearchCase(caseIdentifier = "C5", status = CaseStatus.UNDER_REVIEW))
        repository.insertCase(ResearchCase(caseIdentifier = "C6", status = CaseStatus.REVIEWED))

        val counts = repository.getDashboardCounts()
        assertEquals(6, counts.totalCases)
        assertEquals(2, counts.draftCount)
        assertEquals(1, counts.readyCount)
        assertEquals(1, counts.analyzedCount)
        assertEquals(1, counts.underReviewCount)
        assertEquals(1, counts.reviewedCount)
    }

    @Test
    fun testInferenceRunAssociationIntegrity() {
        val caseAId = repository.insertCase(ResearchCase(caseIdentifier = "CASE-A"))
        val caseBId = repository.insertCase(ResearchCase(caseIdentifier = "CASE-B"))

        // Add run to Case A
        val runA1 = InferenceRun(
            caseId = caseAId,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.9250f,
            logit = 2.512f,
            latencyMs = 38.2,
            numPatches = 50
        )
        val runA1Id = repository.insertInferenceRun(runA1)

        // Add run to Case B
        val runB1 = InferenceRun(
            caseId = caseBId,
            predictedClass = 0,
            predictedLabel = "MSS",
            probability = 0.1240f,
            logit = -1.955f,
            latencyMs = 35.1,
            numPatches = 50
        )
        repository.insertInferenceRun(runB1)

        // Verify runs belong strictly to the correct cases
        val runsA = repository.getRunsForCase(caseAId)
        val runsB = repository.getRunsForCase(caseBId)

        assertEquals(1, runsA.size)
        assertEquals(1, runsB.size)
        assertEquals("MSI", runsA.first().predictedLabel)
        assertEquals("MSS", runsB.first().predictedLabel)
        assertEquals(0.9250f, runsA.first().probability, 1e-4f)
        assertEquals(0.1240f, runsB.first().probability, 1e-4f)

        // Adding second run to Case A maintains run history without overwriting
        val runA2 = InferenceRun(
            caseId = caseAId,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.9410f,
            logit = 2.771f,
            latencyMs = 40.0,
            numPatches = 100,
            executionTimestamp = System.currentTimeMillis() + 1000L
        )
        repository.insertInferenceRun(runA2)

        val updatedRunsA = repository.getRunsForCase(caseAId)
        assertEquals(2, updatedRunsA.size)
        assertEquals(0.9410f, repository.getLatestRunForCase(caseAId)?.probability ?: 0f, 1e-4f)
    }

    @Test
    fun testCaseLinkedReviewNotes() {
        val caseId = repository.insertCase(ResearchCase(caseIdentifier = "CASE-NOTES-01"))

        val note1 = ReviewNote(
            caseId = caseId,
            category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
            authorLabel = "Pathologist A",
            content = "Prominent Crohn's-like lymphoid reaction"
        )
        val note1Id = repository.insertReviewNote(note1)

        val note2 = ReviewNote(
            caseId = caseId,
            category = ReviewNoteCategory.MODEL_UNCERTAINTY,
            authorLabel = "Pathologist B",
            content = "Borderline mucinous component, evaluate further"
        )
        repository.insertReviewNote(note2)

        val notes = repository.getNotesForCase(caseId)
        assertEquals(2, notes.size)

        // Delete note
        assertTrue(repository.deleteReviewNote(note1Id))
        val notesAfterDelete = repository.getNotesForCase(caseId)
        assertEquals(1, notesAfterDelete.size)
        assertEquals(ReviewNoteCategory.MODEL_UNCERTAINTY, notesAfterDelete.first().category)
    }

    @Test
    fun testCascadingDeletion() {
        val caseId = repository.insertCase(ResearchCase(caseIdentifier = "CASE-TO-DELETE"))
        repository.insertSpecimenInput(SpecimenInput(caseId = caseId, inputType = "TILES", localUriOrAsset = "ref", filename = "t1.png"))
        repository.insertInferenceRun(InferenceRun(caseId = caseId, predictedClass = 0, predictedLabel = "MSS", probability = 0.2f, logit = -1.3f, latencyMs = 30.0, numPatches = 10))
        repository.insertReviewNote(ReviewNote(caseId = caseId, content = "Note content"))

        // Delete case
        assertTrue(repository.deleteCase(caseId))
        assertNull(repository.getCaseById(caseId))
        assertTrue(repository.getInputsForCase(caseId).isEmpty())
        assertTrue(repository.getRunsForCase(caseId).isEmpty())
        assertTrue(repository.getNotesForCase(caseId).isEmpty())
    }

    @Test
    fun testReportGenerationFromActualData() {
        val testCase = ResearchCase(
            caseIdentifier = "CRC-REPORT-001",
            specimenDescription = "Right colon biopsy FFPE",
            researchContext = "Elderly female presenting with anemia",
            status = CaseStatus.ANALYZED,
            patchCount = 100,
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED
        )
        val testRun = InferenceRun(
            caseId = 1L,
            modelIdentifier = "fusion_model_fp32.onnx",
            inferenceArchitecture = "LOCAL_ONNX_FP32",
            protocol = InferenceProtocol.PROTOCOL_1_CAPPED,
            predictedClass = 1,
            predictedLabel = "MSI",
            probability = 0.9412f,
            logit = 2.7725f,
            threshold = 0.83f,
            margin = 0.1112f,
            latencyMs = 42.5,
            numPatches = 100,
            status = "SUCCESS"
        )
        val testNotes = listOf(
            ReviewNote(
                caseId = 1L,
                category = ReviewNoteCategory.HISTOLOGICAL_OBSERVATION,
                authorLabel = "Lead Pathologist",
                content = "Marked TIL infiltration and medullary histology."
            )
        )

        val reportText = ReportGenerator.generateTextReport(testCase, testRun, testNotes)

        // Verify deterministic report structure and contents
        assertTrue(reportText.contains("CRC MULTIMODAL PATHOLOGY RESEARCH WORKSTATION REPORT"))
        assertTrue(reportText.contains("CRC-REPORT-001"))
        assertTrue(reportText.contains("Right colon biopsy FFPE"))
        assertTrue(reportText.contains("MSI (Class 1)"))
        assertTrue(reportText.contains("0.94120"))
        assertTrue(reportText.contains("0.8300"))
        assertTrue(reportText.contains("+0.11120 vs 0.8300"))
        assertTrue(reportText.contains("fusion_model_fp32.onnx"))
        assertTrue(reportText.contains("0402F6C522BEAC5733E30EEECAD908927957B5D4164B6B896663F43BA3702D73"))
        assertTrue(reportText.contains("Lead Pathologist"))
        assertTrue(reportText.contains("Marked TIL infiltration and medullary histology."))
        assertTrue(reportText.contains("INVESTIGATIONAL RESEARCH USE ONLY"))
    }

    @Test
    fun testProtocol1AndProtocol2Policies() {
        // Case 1: 541 patches under Protocol 1 passes
        val vision541 = FloatArray(541 * 1024) { 0.01f }
        InputValidator.validateVision(vision541, 541, InferenceProtocol.PROTOCOL_1_CAPPED)

        // Case 2: 1,000 patches under Protocol 1 passes
        val vision1000 = FloatArray(1000 * 1024) { 0.01f }
        InputValidator.validateVision(vision1000, 1000, InferenceProtocol.PROTOCOL_1_CAPPED)

        // Case 3: 1,001 patches under Protocol 1 is strictly rejected
        val vision1001 = FloatArray(1001 * 1024) { 0.01f }
        try {
            InputValidator.validateVision(vision1001, 1001, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected Protocol 1 to reject N > 1,000")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("1000"))
        }

        // Case 4: 2,296 patches under Protocol 1 is strictly rejected
        val vision2296 = FloatArray(2296 * 1024) { 0.01f }
        try {
            InputValidator.validateVision(vision2296, 2296, InferenceProtocol.PROTOCOL_1_CAPPED)
            fail("Expected Protocol 1 to reject 2,296 patches")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("1000"))
        }

        // Case 5: 2,296 patches under Protocol 2 is accepted without truncation
        InputValidator.validateVision(vision2296, 2296, InferenceProtocol.PROTOCOL_2_UNCAPPED)
    }

    @Test
    fun testResearchContextLengthBoundaries() {
        val testLengths = listOf(0, 1000, 1001, 10000, 49999, 50000)
        for (len in testLengths) {
            val text = "A".repeat(len)
            val case = ResearchCase(
                caseIdentifier = "TEST-LEN-$len",
                specimenDescription = "Boundary length test",
                researchContext = text
            )
            val id = repository.insertCase(case)
            val retrieved = repository.getCaseById(id)
            assertNotNull(retrieved)
            assertEquals(len, retrieved?.researchContext?.length)
            assertEquals(text, retrieved?.researchContext)
        }

        // Test 50,001 characters: strictly rejected
        val invalidText = "X".repeat(50001)
        try {
            ResearchCase(
                caseIdentifier = "TEST-LEN-OVERFLOW",
                specimenDescription = "Exceeds 50k",
                researchContext = invalidText
            )
            fail("Expected IllegalArgumentException for 50,001 characters")
        } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("50,000"))
        }
    }

    @Test
    fun testThresholdBoundaryDecisions() {
        // Strict boundary: 0.830000 -> MSI (1), 0.829999 -> MSS (0)
        assertEquals(0, ThresholdLogic.classify(0.829999f))
        assertEquals("MSS", ThresholdLogic.labelFor(0))

        assertEquals(1, ThresholdLogic.classify(0.830000f))
        assertEquals("MSI", ThresholdLogic.labelFor(1))

        assertEquals(1, ThresholdLogic.classify(0.830001f))
        assertEquals("MSI", ThresholdLogic.labelFor(1))
    }
}
