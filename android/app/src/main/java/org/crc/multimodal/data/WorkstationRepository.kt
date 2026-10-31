package org.crc.multimodal.data

data class DashboardCounts(
    val totalCases: Int,
    val draftCount: Int,
    val readyCount: Int,
    val analyzedCount: Int,
    val underReviewCount: Int,
    val reviewedCount: Int
)

/**
 * Repository interface governing all local persistence operations for the pathology research workstation.
 */
interface WorkstationRepository {
    fun insertCase(researchCase: ResearchCase): Long
    fun updateCase(researchCase: ResearchCase): Boolean
    fun deleteCase(caseId: Long): Boolean
    fun getCaseById(id: Long): ResearchCase?
    fun getCaseByIdentifier(identifier: String): ResearchCase?
    fun getAllCases(): List<ResearchCase>
    fun getCasesFiltered(
        status: CaseStatus? = null,
        searchQuery: String? = null,
        sortBy: CaseSortOption = CaseSortOption.DATE_NEWEST
    ): List<ResearchCase>
    fun getRecentCases(limit: Int = 3): List<ResearchCase>
    fun getDashboardCounts(): DashboardCounts

    // Inputs
    fun insertSpecimenInput(input: SpecimenInput): Long
    fun getInputsForCase(caseId: Long): List<SpecimenInput>
    fun deleteInput(inputId: Long): Boolean

    // Inference Runs
    fun insertInferenceRun(run: InferenceRun): Long
    fun getRunsForCase(caseId: Long): List<InferenceRun>
    fun getLatestRunForCase(caseId: Long): InferenceRun?
    fun getAllInferenceRuns(): List<InferenceRun>
    fun recordInferenceResult(caseId: Long, run: InferenceRun, newStatus: CaseStatus = CaseStatus.ANALYZED): Long
    fun recordInferenceFailure(caseId: Long, errorMessage: String)
    fun recoverInterruptedCases()
    fun getResearchAnalyticsSummary(batchRepo: org.crc.multimodal.batch.BatchRepository? = null): org.crc.multimodal.analytics.ResearchAnalyticsSummary
    fun getUnifiedPredictionRecords(batchRepo: org.crc.multimodal.batch.BatchRepository? = null): List<org.crc.multimodal.analytics.SpecimenPredictionRecord>

    // Review Notes
    fun insertReviewNote(note: ReviewNote): Long
    fun updateReviewNote(note: ReviewNote): Boolean
    fun deleteReviewNote(noteId: Long): Boolean
    fun getNotesForCase(caseId: Long): List<ReviewNote>

    // Reports
    fun insertOrUpdateReport(report: ResearchReport): Long
    fun getReportForCase(caseId: Long): ResearchReport?

    // Report History (WP7.6 / Feature C)
    fun insertReportRecord(record: PersistentReportRecord): Long
    fun updateReportRecordStatus(
        reportId: String,
        status: ReportExportStatus,
        fileUri: String? = null,
        filePath: String? = null,
        errorMessage: String? = null
    ): Boolean
    fun getReportHistoryForCase(caseId: Long): List<PersistentReportRecord>
    fun getReportRecordById(reportId: String): PersistentReportRecord?
    fun deleteReportRecord(reportId: String): Boolean

    // Reference Labels & Cohort Evaluation (WP7.4)
    fun insertReferenceLabel(label: org.crc.multimodal.evaluation.ReferenceLabel): Long
    fun insertReferenceLabels(labels: List<org.crc.multimodal.evaluation.ReferenceLabel>): Int
    fun getReferenceLabel(specimenId: String): org.crc.multimodal.evaluation.ReferenceLabel?
    fun getAllReferenceLabels(): List<org.crc.multimodal.evaluation.ReferenceLabel>
    fun deleteReferenceLabel(specimenId: String): Boolean
    fun clearReferenceLabels(): Int
    fun evaluateCohort(
        cohortId: String? = null,
        batchRepo: org.crc.multimodal.batch.BatchRepository? = null
    ): org.crc.multimodal.evaluation.CohortEvaluationSummary

    // Security & Audit Trail (WP7.5)
    fun recordAuditEvent(event: org.crc.multimodal.security.AuditTrailRecord): Long
    fun getAuditTrail(limit: Int = 100): List<org.crc.multimodal.security.AuditTrailRecord>
    fun getAuditTrailForSpecimen(specimenId: String): List<org.crc.multimodal.security.AuditTrailRecord>
    fun clearAuditTrail(): Int

    // Maintenance
    fun clearAllData()
    fun seedReferenceCases()
}

enum class CaseSortOption {
    DATE_NEWEST,
    DATE_OLDEST,
    CASE_IDENTIFIER
}
