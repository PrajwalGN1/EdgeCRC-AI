package org.crc.multimodal

import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.data.PersistentReportRecord
import org.crc.multimodal.data.ReportExportStatus
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.util.UUID

class ReportPersistenceMigrationTest {

    private lateinit var repository: InMemoryWorkstationRepository
    private var testCaseId: Long = 0L

    @Before
    fun setUp() {
        repository = InMemoryWorkstationRepository()
        testCaseId = repository.insertCase(
            ResearchCase(
                caseIdentifier = "TCGA-REP-TEST",
                specimenDescription = "Test biopsy",
                researchContext = "Report persistence verification",
                status = CaseStatus.ANALYZED,
                patchCount = 100,
                protocol = InferenceProtocol.PROTOCOL_1_CAPPED
            )
        )
    }

    @Test
    fun testInsertAndRetrieveSingleReportRecord() {
        val reportId = UUID.randomUUID().toString()
        val record = PersistentReportRecord(
            reportId = reportId,
            caseId = testCaseId,
            inferenceRunId = 55L,
            templateVersion = "1.0.0",
            createdAt = 1710000000000L,
            exportStatus = ReportExportStatus.GENERATED,
            filePath = "/data/user/0/org.crc.multimodal/files/reports/$reportId.pdf",
            sha256Hash = "0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73"
        )

        val insertedId = repository.insertReportRecord(record)
        assertTrue(insertedId > 0L)

        val retrieved = repository.getReportRecordById(reportId)
        assertNotNull(retrieved)
        assertEquals(reportId, retrieved?.reportId)
        assertEquals(testCaseId, retrieved?.caseId)
        assertEquals(55L, retrieved?.inferenceRunId)
        assertEquals("1.0.0", retrieved?.templateVersion)
        assertEquals(ReportExportStatus.GENERATED, retrieved?.exportStatus)
        assertEquals("0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73", retrieved?.sha256Hash)
    }

    @Test
    fun testMultipleReportsPerCaseHistory() {
        val id1 = UUID.randomUUID().toString()
        val id2 = UUID.randomUUID().toString()

        val record1 = PersistentReportRecord(
            reportId = id1,
            caseId = testCaseId,
            inferenceRunId = 1L,
            createdAt = 1000L,
            exportStatus = ReportExportStatus.GENERATED
        )
        val record2 = PersistentReportRecord(
            reportId = id2,
            caseId = testCaseId,
            inferenceRunId = 2L,
            createdAt = 2000L,
            exportStatus = ReportExportStatus.EXPORTED,
            fileUri = "content://com.android.providers.downloads/1"
        )

        repository.insertReportRecord(record1)
        repository.insertReportRecord(record2)

        val history = repository.getReportHistoryForCase(testCaseId)
        assertEquals(2, history.size)
        // Should be sorted by createdAt descending
        assertEquals(id2, history[0].reportId)
        assertEquals(id1, history[1].reportId)
    }

    @Test
    fun testUpdateReportExportStatusAndError() {
        val reportId = UUID.randomUUID().toString()
        val record = PersistentReportRecord(
            reportId = reportId,
            caseId = testCaseId,
            exportStatus = ReportExportStatus.GENERATED
        )
        repository.insertReportRecord(record)

        val updatedSuccess = repository.updateReportRecordStatus(
            reportId = reportId,
            status = ReportExportStatus.EXPORTED,
            fileUri = "content://media/external/downloads/test.pdf"
        )
        assertTrue(updatedSuccess)

        var retrieved = repository.getReportRecordById(reportId)
        assertEquals(ReportExportStatus.EXPORTED, retrieved?.exportStatus)
        assertEquals("content://media/external/downloads/test.pdf", retrieved?.fileUri)

        // Simulate failure update
        repository.updateReportRecordStatus(
            reportId = reportId,
            status = ReportExportStatus.FAILED,
            errorMessage = "Storage permission denied"
        )
        retrieved = repository.getReportRecordById(reportId)
        assertEquals(ReportExportStatus.FAILED, retrieved?.exportStatus)
        assertEquals("Storage permission denied", retrieved?.errorMessage)
    }

    @Test
    fun testDeleteCaseCleansUpAssociatedReportHistory() {
        val reportId = UUID.randomUUID().toString()
        repository.insertReportRecord(
            PersistentReportRecord(
                reportId = reportId,
                caseId = testCaseId
            )
        )

        assertEquals(1, repository.getReportHistoryForCase(testCaseId).size)

        repository.deleteCase(testCaseId)

        assertEquals(0, repository.getReportHistoryForCase(testCaseId).size)
        assertNull(repository.getReportRecordById(reportId))
    }

    @Test
    fun testClearAllDataWipesReportHistory() {
        repository.insertReportRecord(
            PersistentReportRecord(
                reportId = UUID.randomUUID().toString(),
                caseId = testCaseId
            )
        )

        repository.clearAllData()

        assertEquals(0, repository.getReportHistoryForCase(testCaseId).size)
    }
}
