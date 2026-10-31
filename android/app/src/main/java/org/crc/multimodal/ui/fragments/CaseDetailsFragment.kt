package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.ui.WorkstationNavigationHost
import org.crc.multimodal.ui.adapters.InferenceRunAdapter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class CaseDetailsFragment : Fragment(R.layout.fragment_case_details) {

    private var navHost: WorkstationNavigationHost? = null
    private var caseId: Long = 0L
    private lateinit var runAdapter: InferenceRunAdapter

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        caseId = arguments?.getLong(ARG_CASE_ID) ?: 0L
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "Case Details",
            subtitle = "Specimen Profile & Run History",
            showBackButton = true
        )

        val host = navHost ?: return
        val repository = host.getRepository()
        val currentCase = repository.getCaseById(caseId) ?: run {
            host.navigateToWorkspace()
            return
        }

        val tvDetailCaseId: TextView = view.findViewById(R.id.tvDetailCaseId)
        val tvDetailCaseStatus: TextView = view.findViewById(R.id.tvDetailCaseStatus)
        val tvDetailSpecimen: TextView = view.findViewById(R.id.tvDetailSpecimen)
        val tvDetailResearchContext: TextView = view.findViewById(R.id.tvDetailResearchContext)
        val tvDetailPatchesAndProtocol: TextView = view.findViewById(R.id.tvDetailPatchesAndProtocol)
        val tvDetailModifiedDate: TextView = view.findViewById(R.id.tvDetailModifiedDate)

        val btnDetailRunInference: MaterialButton = view.findViewById(R.id.btnDetailRunInference)
        val btnDetailViewFindings: MaterialButton = view.findViewById(R.id.btnDetailViewFindings)
        val btnDetailManageNotes: MaterialButton = view.findViewById(R.id.btnDetailManageNotes)
        val btnDetailViewReport: MaterialButton = view.findViewById(R.id.btnDetailViewReport)
        val btnDetailDeleteCase: MaterialButton = view.findViewById(R.id.btnDetailDeleteCase)

        val layoutDetailNoRuns: LinearLayout = view.findViewById(R.id.layoutDetailNoRuns)
        val rvDetailRunsHistory: RecyclerView = view.findViewById(R.id.rvDetailRunsHistory)

        // Populate Case Info
        tvDetailCaseId.text = currentCase.caseIdentifier
        tvDetailCaseStatus.text = currentCase.status.displayName

        when (currentCase.status) {
            CaseStatus.DRAFT -> {
                tvDetailCaseStatus.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_warning_bg)
                tvDetailCaseStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_warning))
            }
            CaseStatus.READY_FOR_ANALYSIS -> {
                tvDetailCaseStatus.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.teal_light)
                tvDetailCaseStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.secondary))
            }
            CaseStatus.ANALYZED -> {
                tvDetailCaseStatus.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_success_bg)
                tvDetailCaseStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_success))
            }
            CaseStatus.UNDER_REVIEW -> {
                tvDetailCaseStatus.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_pending_bg)
                tvDetailCaseStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_pending))
            }
            CaseStatus.REVIEWED -> {
                tvDetailCaseStatus.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.teal_light)
                tvDetailCaseStatus.setTextColor(ContextCompat.getColor(requireContext(), R.color.teal_dark))
            }
        }

        tvDetailSpecimen.text = currentCase.specimenDescription.ifBlank { "No specimen description recorded" }
        tvDetailResearchContext.text = currentCase.researchContext.ifBlank { "No research context recorded" }

        val protocolName = InferenceProtocol.formatProtocol(currentCase.protocol)
        tvDetailPatchesAndProtocol.text = "${currentCase.patchCount} patches · $protocolName"

        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)
        tvDetailModifiedDate.text = "Updated: ${dateFormat.format(Date(currentCase.updatedAt))}"

        // Runs History RecyclerView
        rvDetailRunsHistory.layoutManager = LinearLayoutManager(requireContext())
        runAdapter = InferenceRunAdapter(emptyList())
        rvDetailRunsHistory.adapter = runAdapter

        val runs = repository.getRunsForCase(caseId)
        if (runs.isEmpty()) {
            layoutDetailNoRuns.visibility = View.VISIBLE
            btnDetailViewFindings.isEnabled = false
            btnDetailViewReport.isEnabled = false
        } else {
            layoutDetailNoRuns.visibility = View.GONE
            runAdapter.submitList(runs)
            btnDetailViewFindings.isEnabled = true
            btnDetailViewReport.isEnabled = true
        }

        // Actions
        btnDetailRunInference.setOnClickListener {
            host.navigateToInferenceExecution(currentCase.id)
        }

        btnDetailViewFindings.setOnClickListener {
            host.navigateToFindings(currentCase.id)
        }

        btnDetailManageNotes.setOnClickListener {
            host.navigateToReviewNotes(currentCase.id)
        }

        btnDetailViewReport.setOnClickListener {
            host.navigateToReport(currentCase.id)
        }

        btnDetailDeleteCase.setOnClickListener {
            MaterialAlertDialogBuilder(requireContext())
                .setTitle("Delete Case Record?")
                .setMessage("Are you sure you want to permanently delete case ${currentCase.caseIdentifier} and all associated runs and notes?")
                .setPositiveButton("Delete") { _, _ ->
                    repository.deleteCase(currentCase.id)
                    Toast.makeText(requireContext(), "Case record deleted", Toast.LENGTH_SHORT).show()
                    host.navigateToCases()
                }
                .setNegativeButton("Cancel", null)
                .show()
        }
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"

        fun newInstance(caseId: Long): CaseDetailsFragment {
            return CaseDetailsFragment().apply {
                arguments = Bundle().apply {
                    putLong(ARG_CASE_ID, caseId)
                }
            }
        }
    }
}
