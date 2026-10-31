package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.inference.OnnxRuntimeState
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.CaseAdapter

class WorkspaceFragment : Fragment(R.layout.fragment_workspace) {

    private var navHost: WorkstationNavigationHost? = null
    private lateinit var caseAdapter: CaseAdapter
    private var runtimeStateListener: ((OnnxRuntimeState) -> Unit)? = null

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = getString(R.string.title_workstation),
            subtitle = getString(R.string.subtitle_workstation),
            showBackButton = false
        )

        val host = navHost ?: return
        val repository = host.getRepository()

        // Bind Metrics Views
        val tvMetricTotalCases: TextView = view.findViewById(R.id.tvMetricTotalCases)
        val tvMetricDraftCount: TextView = view.findViewById(R.id.tvMetricDraftCount)
        val tvMetricAnalyzedCount: TextView = view.findViewById(R.id.tvMetricAnalyzedCount)
        val tvMetricUnderReviewCount: TextView = view.findViewById(R.id.tvMetricUnderReviewCount)

        // Engine Status Card & Lifecycle-Safe State Observation
        val cardWorkspaceRuntime: MaterialCardView? = view.findViewById(R.id.cardWorkspaceRuntime)
        cardWorkspaceRuntime?.setOnClickListener {
            val currentState = host.getOnnxRuntimeState()
            if (currentState is OnnxRuntimeState.Failed) {
                host.retryClassifierInitialization()
            }
        }

        val listener: (OnnxRuntimeState) -> Unit = { state ->
            view?.let { v ->
                renderEngineStatus(v, state)
            }
        }
        runtimeStateListener = listener
        host.addOnnxRuntimeStateListener(listener)

        // Action Buttons
        val btnWorkspaceNewSpecimen: MaterialButton = view.findViewById(R.id.btnWorkspaceNewSpecimen)
        btnWorkspaceNewSpecimen.setOnClickListener {
            host.navigateToNewSpecimen()
        }

        val btnWorkspaceBatchAnalysis: MaterialButton? = view.findViewById(R.id.btnWorkspaceBatchAnalysis)
        btnWorkspaceBatchAnalysis?.setOnClickListener {
            host.navigateToBatchAnalysis()
        }

        val btnWorkspaceAnalytics: MaterialButton? = view.findViewById(R.id.btnWorkspaceAnalytics)
        btnWorkspaceAnalytics?.setOnClickListener {
            host.navigateToAnalyticsDashboard()
        }

        val btnWorkspaceCohortEvaluation: MaterialButton? = view.findViewById(R.id.btnWorkspaceCohortEvaluation)
        btnWorkspaceCohortEvaluation?.setOnClickListener {
            host.navigateToCohortEvaluation()
        }

        val tvWorkspaceViewAllCases: TextView = view.findViewById(R.id.tvWorkspaceViewAllCases)
        tvWorkspaceViewAllCases.setOnClickListener {
            host.navigateToCases()
        }

        val layoutWorkspaceEmpty: LinearLayout = view.findViewById(R.id.layoutWorkspaceEmpty)
        val btnWorkspaceSeedReferenceCases: MaterialButton = view.findViewById(R.id.btnWorkspaceSeedReferenceCases)
        btnWorkspaceSeedReferenceCases.setOnClickListener {
            repository.seedReferenceCases()
            refreshDashboard(tvMetricTotalCases, tvMetricDraftCount, tvMetricAnalyzedCount, tvMetricUnderReviewCount, layoutWorkspaceEmpty)
        }

        // Setup Recent Cases RecyclerView
        val rvWorkspaceRecentCases: RecyclerView = view.findViewById(R.id.rvWorkspaceRecentCases)
        rvWorkspaceRecentCases.layoutManager = LinearLayoutManager(requireContext())
        caseAdapter = CaseAdapter(emptyList(), repository) { selectedCase ->
            host.navigateToCaseDetails(selectedCase.id)
        }
        rvWorkspaceRecentCases.adapter = caseAdapter

        // Load & Render Dashboard Data
        refreshDashboard(tvMetricTotalCases, tvMetricDraftCount, tvMetricAnalyzedCount, tvMetricUnderReviewCount, layoutWorkspaceEmpty)
    }

    private fun refreshDashboard(
        tvTotal: TextView,
        tvDraft: TextView,
        tvAnalyzed: TextView,
        tvUnderReview: TextView,
        emptyLayout: LinearLayout
    ) {
        val host = navHost ?: return
        val repository = host.getRepository()

        val counts = repository.getDashboardCounts()
        tvTotal.text = counts.totalCases.toString()
        tvDraft.text = counts.draftCount.toString()
        tvAnalyzed.text = counts.analyzedCount.toString()
        tvUnderReview.text = counts.underReviewCount.toString()

        val recentCases = repository.getRecentCases(limit = 4)
        if (recentCases.isEmpty()) {
            emptyLayout.visibility = View.VISIBLE
            caseAdapter.submitList(emptyList())
        } else {
            emptyLayout.visibility = View.GONE
            caseAdapter.submitList(recentCases)
        }
    }

    override fun onResume() {
        super.onResume()
        view?.let { v ->
            val tvTotal: TextView = v.findViewById(R.id.tvMetricTotalCases)
            val tvDraft: TextView = v.findViewById(R.id.tvMetricDraftCount)
            val tvAnalyzed: TextView = v.findViewById(R.id.tvMetricAnalyzedCount)
            val tvUnderReview: TextView = v.findViewById(R.id.tvMetricUnderReviewCount)
            val emptyLayout: LinearLayout = v.findViewById(R.id.layoutWorkspaceEmpty)
            refreshDashboard(tvTotal, tvDraft, tvAnalyzed, tvUnderReview, emptyLayout)

            navHost?.let { host ->
                renderEngineStatus(v, host.getOnnxRuntimeState())
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        runtimeStateListener?.let { navHost?.removeOnnxRuntimeStateListener(it) }
        runtimeStateListener = null
    }

    private fun renderEngineStatus(rootView: View, state: OnnxRuntimeState) {
        val ivWorkspaceEngineStatus: ImageView = rootView.findViewById(R.id.ivWorkspaceEngineStatus) ?: return
        val tvWorkspaceEngineTitle: TextView = rootView.findViewById(R.id.tvWorkspaceEngineTitle) ?: return
        val tvWorkspaceEngineSubtitle: TextView = rootView.findViewById(R.id.tvWorkspaceEngineSubtitle) ?: return
        val context = rootView.context ?: return

        when (state) {
            is OnnxRuntimeState.NotStarted -> {
                ivWorkspaceEngineStatus.setImageResource(R.drawable.ic_speed)
                ivWorkspaceEngineStatus.setColorFilter(ContextCompat.getColor(context, R.color.text_supporting))
                tvWorkspaceEngineTitle.text = "Inference Engine: Standby"
                tvWorkspaceEngineSubtitle.text = "Awaiting model session startup"
            }
            is OnnxRuntimeState.Initializing -> {
                ivWorkspaceEngineStatus.setImageResource(R.drawable.ic_speed)
                ivWorkspaceEngineStatus.setColorFilter(ContextCompat.getColor(context, R.color.status_pending))
                tvWorkspaceEngineTitle.text = "ONNX Runtime Initializing…"
                tvWorkspaceEngineSubtitle.text = "Loading model graph & creating CPU session…"
            }
            is OnnxRuntimeState.Ready -> {
                ivWorkspaceEngineStatus.setImageResource(R.drawable.ic_check_circle)
                ivWorkspaceEngineStatus.setColorFilter(ContextCompat.getColor(context, R.color.status_success))
                tvWorkspaceEngineTitle.text = "FP32 ONNX Runtime Engine: Ready"
                tvWorkspaceEngineSubtitle.text = "CPUExecutionProvider · Fixed Threshold: 0.83 · On-device Session Active"
            }
            is OnnxRuntimeState.Failed -> {
                ivWorkspaceEngineStatus.setImageResource(R.drawable.ic_info_outline)
                ivWorkspaceEngineStatus.setColorFilter(ContextCompat.getColor(context, R.color.status_error))
                tvWorkspaceEngineTitle.text = "Inference Engine: Initialization Failed"
                tvWorkspaceEngineSubtitle.text = "${state.errorMessage} (Tap card to retry)"
            }
        }
    }
}
