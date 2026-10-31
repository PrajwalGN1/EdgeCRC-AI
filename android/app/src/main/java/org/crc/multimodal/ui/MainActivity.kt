package org.crc.multimodal.ui

import android.os.Bundle
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.bottomnavigation.BottomNavigationView
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.SqliteWorkstationRepository
import org.crc.multimodal.data.WorkstationRepository
import org.crc.multimodal.inference.CrcMultimodalClassifier
import org.crc.multimodal.ui.fragments.CaseDetailsFragment
import org.crc.multimodal.ui.fragments.CasesFragment
import org.crc.multimodal.ui.fragments.FindingsFragment
import org.crc.multimodal.ui.fragments.InferenceExecutionFragment
import org.crc.multimodal.ui.fragments.NewSpecimenFragment
import org.crc.multimodal.ui.fragments.ReportFragment
import org.crc.multimodal.ui.fragments.ReviewNotesFragment
import org.crc.multimodal.ui.fragments.SettingsFragment
import org.crc.multimodal.ui.fragments.SpecimenReviewFragment
import org.crc.multimodal.ui.fragments.WorkspaceFragment
import org.crc.multimodal.inference.CrcLogger
import org.crc.multimodal.inference.OnnxRuntimeManager
import org.crc.multimodal.inference.OnnxRuntimeState

/**
 * Enterprise-Grade Multimodal Edge AI Pathology Research Workstation.
 *
 * Hosts multi-screen navigation across 10 specialized workflow destinations:
 * 1. Research Workspace Dashboard
 * 2. Case Management
 * 3. New Specimen Analysis Intake
 * 4. Specimen Review Verification
 * 5. Inference Configuration & Execution
 * 6. Analysis Findings
 * 7. Review Questions & Notes
 * 8. Structured Research Report
 * 9. Case Details & Run History
 * 10. Settings & Runtime Specifications
 */
class MainActivity : AppCompatActivity(), WorkstationNavigationHost {

    private lateinit var toolbar: MaterialToolbar
    private lateinit var bottomNav: BottomNavigationView
    private lateinit var repository: WorkstationRepository
    private lateinit var batchRepository: org.crc.multimodal.batch.BatchRepository

    private val onnxRuntimeManager: OnnxRuntimeManager = OnnxRuntimeManager.getInstance()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        repository = SqliteWorkstationRepository.getInstance(applicationContext)
        repository.seedReferenceCases()
        repository.recoverInterruptedCases()

        batchRepository = org.crc.multimodal.batch.SqliteBatchRepository.getInstance(applicationContext)
        batchRepository.recoverInterruptedBatches()

        toolbar = findViewById(R.id.topToolbar)
        bottomNav = findViewById(R.id.bottom_navigation)

        setupToolbarNavigation()
        setupBottomNavigation()
        setupBackstackListener()
        setupSystemBackDispatcher()

        onnxRuntimeManager.initializeAsync(applicationContext)

        if (savedInstanceState == null) {
            navigateToWorkspace()
        }
    }

    private fun setupToolbarNavigation() {
        toolbar.setNavigationOnClickListener {
            if (supportFragmentManager.backStackEntryCount > 0) {
                supportFragmentManager.popBackStack()
            }
        }
    }

    private fun setupBottomNavigation() {
        bottomNav.setOnItemSelectedListener { item ->
            when (item.itemId) {
                R.id.nav_workspace -> {
                    clearBackStack()
                    replaceRootFragment(WorkspaceFragment())
                    true
                }
                R.id.nav_cases -> {
                    clearBackStack()
                    replaceRootFragment(CasesFragment())
                    true
                }
                R.id.nav_new_analysis -> {
                    clearBackStack()
                    replaceRootFragment(NewSpecimenFragment.newInstance())
                    true
                }
                R.id.nav_settings -> {
                    clearBackStack()
                    replaceRootFragment(SettingsFragment())
                    true
                }
                else -> false
            }
        }
    }

    private fun setupBackstackListener() {
        supportFragmentManager.addOnBackStackChangedListener {
            val backStackCount = supportFragmentManager.backStackEntryCount
            if (backStackCount == 0) {
                // Return to root bottom nav state
                val currentFragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
                when (currentFragment) {
                    is WorkspaceFragment -> bottomNav.menu.findItem(R.id.nav_workspace)?.isChecked = true
                    is CasesFragment -> bottomNav.menu.findItem(R.id.nav_cases)?.isChecked = true
                    is NewSpecimenFragment -> bottomNav.menu.findItem(R.id.nav_new_analysis)?.isChecked = true
                    is SettingsFragment -> bottomNav.menu.findItem(R.id.nav_settings)?.isChecked = true
                }
            }
        }
    }

    private fun setupSystemBackDispatcher() {
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (supportFragmentManager.backStackEntryCount > 0) {
                    supportFragmentManager.popBackStack()
                } else {
                    val currentFragment = supportFragmentManager.findFragmentById(R.id.fragment_container)
                    if (currentFragment !is WorkspaceFragment) {
                        bottomNav.selectedItemId = R.id.nav_workspace
                    } else {
                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                    }
                }
            }
        })
    }

    private fun replaceRootFragment(fragment: Fragment) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .commit()
    }

    private fun pushFragment(fragment: Fragment) {
        supportFragmentManager.beginTransaction()
            .replace(R.id.fragment_container, fragment)
            .addToBackStack(null)
            .commit()
    }

    private fun clearBackStack() {
        if (supportFragmentManager.backStackEntryCount > 0) {
            supportFragmentManager.popBackStack(null, FragmentManager.POP_BACK_STACK_INCLUSIVE)
        }
    }

    // WorkstationNavigationHost implementation
    override fun navigateToWorkspace() {
        clearBackStack()
        bottomNav.selectedItemId = R.id.nav_workspace
    }

    override fun navigateToCases(filterStatus: CaseStatus?) {
        clearBackStack()
        bottomNav.selectedItemId = R.id.nav_cases
    }

    override fun navigateToNewSpecimen(caseId: Long?) {
        pushFragment(NewSpecimenFragment.newInstance(caseId))
    }

    override fun navigateToSpecimenReview(caseId: Long) {
        pushFragment(SpecimenReviewFragment.newInstance(caseId))
    }

    override fun navigateToInferenceExecution(caseId: Long) {
        pushFragment(InferenceExecutionFragment.newInstance(caseId))
    }

    override fun navigateToFindings(caseId: Long, runId: Long?) {
        pushFragment(FindingsFragment.newInstance(caseId, runId))
    }

    override fun navigateToReviewNotes(caseId: Long) {
        pushFragment(ReviewNotesFragment.newInstance(caseId))
    }

    override fun navigateToReport(caseId: Long, runId: Long?) {
        pushFragment(ReportFragment.newInstance(caseId, runId))
    }

    override fun navigateToCaseDetails(caseId: Long) {
        pushFragment(CaseDetailsFragment.newInstance(caseId))
    }

    override fun navigateToSettings() {
        clearBackStack()
        bottomNav.selectedItemId = R.id.nav_settings
    }

    override fun navigateToBatchAnalysis(batchId: String?) {
        pushFragment(org.crc.multimodal.ui.fragments.BatchAnalysisFragment.newInstance(batchId))
    }

    override fun navigateToAnalyticsDashboard() {
        pushFragment(org.crc.multimodal.ui.fragments.AnalyticsDashboardFragment.newInstance())
    }

    override fun navigateToCohortEvaluation(cohortId: String?) {
        pushFragment(org.crc.multimodal.ui.fragments.CohortEvaluationFragment.newInstance(cohortId))
    }

    override fun getRepository(): WorkstationRepository = repository

    override fun getBatchRepository(): org.crc.multimodal.batch.BatchRepository = batchRepository

    override fun getClassifier(): CrcMultimodalClassifier? = onnxRuntimeManager.classifier

    override fun isClassifierReady(): Boolean = onnxRuntimeManager.isReady

    override fun getOnnxRuntimeState(): OnnxRuntimeState = onnxRuntimeManager.state

    override fun addOnnxRuntimeStateListener(listener: (OnnxRuntimeState) -> Unit) {
        onnxRuntimeManager.addListener(listener)
    }

    override fun removeOnnxRuntimeStateListener(listener: (OnnxRuntimeState) -> Unit) {
        onnxRuntimeManager.removeListener(listener)
    }

    override fun retryClassifierInitialization() {
        onnxRuntimeManager.initializeAsync(applicationContext, forceRetry = true)
    }

    override fun setTopToolbar(title: String, subtitle: String?, showBackButton: Boolean) {
        toolbar.title = title
        toolbar.subtitle = subtitle
        if (showBackButton) {
            toolbar.setNavigationIcon(R.drawable.ic_arrow_back)
        } else {
            toolbar.setNavigationIcon(R.drawable.ic_science)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isFinishing) {
            onnxRuntimeManager.close()
        }
    }
}
