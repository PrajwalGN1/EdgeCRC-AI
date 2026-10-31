package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import org.crc.multimodal.R
import org.crc.multimodal.data.InferenceRun
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ThresholdLogic
import org.crc.multimodal.ui.ProbabilityThresholdBarView
import org.crc.multimodal.ui.WorkstationNavigationHost
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FindingsFragment : Fragment(R.layout.fragment_findings) {

    private var navHost: WorkstationNavigationHost? = null
    private var caseId: Long = 0L
    private var runId: Long? = null
    private var isDiagnosticsExpanded: Boolean = false

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caseId = arguments?.getLong(ARG_CASE_ID) ?: 0L
        runId = arguments?.getLong(ARG_RUN_ID)?.takeIf { it > 0 }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Analysis Findings",
            subtitle = "Multimodal MSI/MSS Prediction",
            showBackButton = true
        )

        val host = navHost ?: return
        val currentCase = host.getRepository().getCaseById(caseId) ?: run {
            host.navigateToWorkspace()
            return
        }

        val inferenceRun = (runId?.let { host.getRepository().getRunsForCase(caseId).firstOrNull { r -> r.id == it } }
            ?: host.getRepository().getLatestRunForCase(caseId)) ?: run {
            host.navigateToCaseDetails(caseId)
            return
        }

        val tvFindingsCaseId: TextView = view.findViewById(R.id.tvFindingsCaseId)
        val tvFindingsTimestamp: TextView = view.findViewById(R.id.tvFindingsTimestamp)
        val ivFindingsClassificationIcon: ImageView = view.findViewById(R.id.ivFindingsClassificationIcon)
        val tvFindingsClassification: TextView = view.findViewById(R.id.tvFindingsClassification)
        val tvFindingsSubLabel: TextView = view.findViewById(R.id.tvFindingsSubLabel)
        val tvFindingsScore: TextView = view.findViewById(R.id.tvFindingsScore)
        val tvFindingsMargin: TextView = view.findViewById(R.id.tvFindingsMargin)
        val findingsProbabilityBar: ProbabilityThresholdBarView = view.findViewById(R.id.findingsProbabilityBar)

        val tvFindingsLatency: TextView = view.findViewById(R.id.tvFindingsLatency)
        val tvFindingsPatches: TextView = view.findViewById(R.id.tvFindingsPatches)
        val tvFindingsProtocol: TextView = view.findViewById(R.id.tvFindingsProtocol)

        val btnToggleDiagnosticsFindings: MaterialButton = view.findViewById(R.id.btnToggleDiagnosticsFindings)
        val layoutFindingsDiagnostics: LinearLayout = view.findViewById(R.id.layoutFindingsDiagnostics)
        val tvFindingsDiagnosticsText: TextView = view.findViewById(R.id.tvFindingsDiagnosticsText)

        val btnFindingsAddNote: MaterialButton = view.findViewById(R.id.btnFindingsAddNote)
        val btnFindingsGenerateReport: MaterialButton = view.findViewById(R.id.btnFindingsGenerateReport)
        val btnFindingsBackToCase: MaterialButton = view.findViewById(R.id.btnFindingsBackToCase)

        // Case Info
        tvFindingsCaseId.text = currentCase.caseIdentifier
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        tvFindingsTimestamp.text = dateFormat.format(Date(inferenceRun.executionTimestamp))

        // Classification details
        val isMsi = inferenceRun.predictedClass == ThresholdLogic.CLASS_MSI
        if (isMsi) {
            tvFindingsClassification.text = "MSI (Microsatellite Instability)"
            tvFindingsClassification.setTextColor(ContextCompat.getColor(requireContext(), R.color.msi_color))
            tvFindingsSubLabel.text = "Class 1 · Probability exceeds 0.83 decision boundary"
            ivFindingsClassificationIcon.setImageResource(R.drawable.ic_check_circle)
            ivFindingsClassificationIcon.setColorFilter(ContextCompat.getColor(requireContext(), R.color.msi_color))
        } else {
            tvFindingsClassification.text = "MSS (Microsatellite Stable)"
            tvFindingsClassification.setTextColor(ContextCompat.getColor(requireContext(), R.color.mss_color))
            tvFindingsSubLabel.text = "Class 0 · Probability is below 0.83 decision boundary"
            ivFindingsClassificationIcon.setImageResource(R.drawable.ic_check_circle)
            ivFindingsClassificationIcon.setColorFilter(ContextCompat.getColor(requireContext(), R.color.mss_color))
        }

        tvFindingsScore.text = String.format(Locale.US, "Probability: %.4f", inferenceRun.probability)
        tvFindingsMargin.text = if (inferenceRun.margin >= 0) {
            String.format(Locale.US, "+%.4f above 0.83 threshold", inferenceRun.margin)
        } else {
            String.format(Locale.US, "%.4f below 0.83 threshold", inferenceRun.margin)
        }

        // Custom Native Visual Bar
        findingsProbabilityBar.setProbability(inferenceRun.probability, inferenceRun.threshold)

        // Metrics
        tvFindingsLatency.text = String.format(Locale.US, "%.2f ms", inferenceRun.latencyMs)
        tvFindingsPatches.text = "${inferenceRun.numPatches} Patches"
        tvFindingsProtocol.text = InferenceProtocol.formatProtocol(inferenceRun.protocol)

        // Diagnostics
        tvFindingsDiagnosticsText.text = inferenceRun.diagnosticsSummary ?: "No diagnostics log attached."
        btnToggleDiagnosticsFindings.setOnClickListener {
            isDiagnosticsExpanded = !isDiagnosticsExpanded
            if (isDiagnosticsExpanded) {
                layoutFindingsDiagnostics.visibility = View.VISIBLE
                btnToggleDiagnosticsFindings.text = getString(R.string.btn_toggle_diagnostics_hide)
                btnToggleDiagnosticsFindings.setIconResource(R.drawable.ic_expand_less)
            } else {
                layoutFindingsDiagnostics.visibility = View.GONE
                btnToggleDiagnosticsFindings.text = getString(R.string.btn_toggle_diagnostics_show)
                btnToggleDiagnosticsFindings.setIconResource(R.drawable.ic_expand_more)
            }
        }

        btnFindingsAddNote.setOnClickListener {
            host.navigateToReviewNotes(currentCase.id)
        }

        btnFindingsGenerateReport.setOnClickListener {
            host.navigateToReport(currentCase.id, inferenceRun.id)
        }

        btnFindingsBackToCase.setOnClickListener {
            host.navigateToCaseDetails(currentCase.id)
        }
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"
        private const val ARG_RUN_ID = "arg_run_id"

        fun newInstance(caseId: Long, runId: Long? = null): FindingsFragment {
            return FindingsFragment().apply {
                arguments = Bundle().apply {
                    putLong(ARG_CASE_ID, caseId)
                    if (runId != null) putLong(ARG_RUN_ID, runId)
                }
            }
        }
    }
}
