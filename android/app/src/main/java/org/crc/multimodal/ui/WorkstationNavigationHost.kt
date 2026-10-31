package org.crc.multimodal.ui

import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.WorkstationRepository
import org.crc.multimodal.inference.CrcMultimodalClassifier

import org.crc.multimodal.inference.OnnxRuntimeState

/**
 * Navigation interface implemented by MainActivity to manage seamless fragment transactions,
 * backstack navigation, toolbar updates, and shared classifier/repository state.
 */
interface WorkstationNavigationHost {
    fun navigateToWorkspace()
    fun navigateToCases(filterStatus: CaseStatus? = null)
    fun navigateToNewSpecimen(caseId: Long? = null)
    fun navigateToSpecimenReview(caseId: Long)
    fun navigateToInferenceExecution(caseId: Long)
    fun navigateToFindings(caseId: Long, runId: Long? = null)
    fun navigateToReviewNotes(caseId: Long)
    fun navigateToReport(caseId: Long, runId: Long? = null)
    fun navigateToCaseDetails(caseId: Long)
    fun navigateToSettings()
    fun navigateToBatchAnalysis(batchId: String? = null)
    fun navigateToAnalyticsDashboard()
    fun navigateToCohortEvaluation(cohortId: String? = null)

    fun getRepository(): WorkstationRepository
    fun getBatchRepository(): org.crc.multimodal.batch.BatchRepository
    fun getClassifier(): CrcMultimodalClassifier?
    fun isClassifierReady(): Boolean
    fun getOnnxRuntimeState(): OnnxRuntimeState
    fun addOnnxRuntimeStateListener(listener: (OnnxRuntimeState) -> Unit)
    fun removeOnnxRuntimeStateListener(listener: (OnnxRuntimeState) -> Unit)
    fun retryClassifierInitialization()
    fun setTopToolbar(title: String, subtitle: String? = null, showBackButton: Boolean = false)
}
