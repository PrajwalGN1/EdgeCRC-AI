package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import org.crc.multimodal.R
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.BundleStorageManager
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.inference.ReferenceTensorLoader
import org.crc.multimodal.ui.WorkstationNavigationHost

class SpecimenReviewFragment : Fragment(R.layout.fragment_specimen_review) {

    private var navHost: WorkstationNavigationHost? = null
    private var caseId: Long = 0L

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
            title = "Specimen Review",
            subtitle = "Pre-Inference Verification Checklist",
            showBackButton = true
        )

        val host = navHost ?: return
        val currentCase = host.getRepository().getCaseById(caseId) ?: run {
            host.navigateToWorkspace()
            return
        }

        val tvReviewCaseId: TextView = view.findViewById(R.id.tvReviewCaseId)
        val tvReviewStatus: TextView = view.findViewById(R.id.tvReviewStatus)
        val tvReviewSpecimenDescription: TextView = view.findViewById(R.id.tvReviewSpecimenDescription)
        val tvReviewClinicalContext: TextView = view.findViewById(R.id.tvReviewClinicalContext)

        val ivCheckVisionStatus: ImageView = view.findViewById(R.id.ivCheckVisionStatus)
        val tvCheckVisionDetails: TextView = view.findViewById(R.id.tvCheckVisionDetails)

        val ivCheckTextStatus: ImageView = view.findViewById(R.id.ivCheckTextStatus)
        val tvCheckTextDetails: TextView = view.findViewById(R.id.tvCheckTextDetails)

        val ivCheckProtocolStatus: ImageView = view.findViewById(R.id.ivCheckProtocolStatus)
        val tvCheckProtocolDetails: TextView = view.findViewById(R.id.tvCheckProtocolDetails)

        val ivCheckEngineStatus: ImageView = view.findViewById(R.id.ivCheckEngineStatus)
        val tvCheckEngineDetails: TextView = view.findViewById(R.id.tvCheckEngineDetails)

        val btnProceedToExecution: MaterialButton = view.findViewById(R.id.btnProceedToExecution)
        val btnEditSpecimenInputs: MaterialButton = view.findViewById(R.id.btnEditSpecimenInputs)

        // Populate Case Info
        tvReviewCaseId.text = currentCase.caseIdentifier
        tvReviewStatus.text = currentCase.status.displayName
        tvReviewSpecimenDescription.text = currentCase.specimenDescription
        tvReviewClinicalContext.text = currentCase.researchContext.ifBlank { "De-identified clinical research context" }

        val isBenchmark = ReferenceTensorLoader.isReferenceCase(currentCase.caseIdentifier)
        val hasImportedBundle = BundleStorageManager.hasBundle(requireContext(), currentCase.caseIdentifier)
        val isAuthentic = isBenchmark || hasImportedBundle

        // Populate Vision Check
        if (hasImportedBundle) {
            ivCheckVisionStatus.setImageResource(R.drawable.ic_check_circle)
            ivCheckVisionStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_success))
            tvCheckVisionDetails.text = "Imported Authentic [1, ${currentCase.patchCount}, 1024] Float32 tensor · Verified SHA-256"
        } else if (isBenchmark) {
            ivCheckVisionStatus.setImageResource(R.drawable.ic_check_circle)
            ivCheckVisionStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_success))
            tvCheckVisionDetails.text = "Holdout Benchmark [1, ${currentCase.patchCount}, 1024] Float32 tensor · Frozen asset"
        } else {
            ivCheckVisionStatus.setImageResource(R.drawable.ic_error_outline)
            ivCheckVisionStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_error))
            tvCheckVisionDetails.text = "MISSING: No authentic vision tensor. Synthetic vectors forbidden."
        }

        // Populate Text Check
        if (hasImportedBundle) {
            ivCheckTextStatus.setImageResource(R.drawable.ic_check_circle)
            ivCheckTextStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_success))
            tvCheckTextDetails.text = "Imported Authentic [1, 768] Float32 tensor · Verified SHA-256"
        } else if (isBenchmark) {
            ivCheckTextStatus.setImageResource(R.drawable.ic_check_circle)
            ivCheckTextStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_success))
            tvCheckTextDetails.text = "Holdout Benchmark [1, 768] Float32 tensor (ClinicalBERT precomputed)"
        } else {
            ivCheckTextStatus.setImageResource(R.drawable.ic_error_outline)
            ivCheckTextStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_error))
            tvCheckTextDetails.text = "MISSING: No authentic text tensor. Authentic bundle required."
        }

        // Populate Protocol Check
        val protocolText = InferenceProtocol.formatProtocolDetails(currentCase.protocol, currentCase.patchCount)
        tvCheckProtocolDetails.text = protocolText
        val protocolColor = if (currentCase.protocol != null) R.color.status_success else R.color.status_warning
        ivCheckProtocolStatus.setColorFilter(ContextCompat.getColor(requireContext(), protocolColor))

        // Populate Engine Readiness
        val isEngineReady = host.isClassifierReady()
        if (isEngineReady) {
            ivCheckEngineStatus.setImageResource(R.drawable.ic_check_circle)
            ivCheckEngineStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_success))
            tvCheckEngineDetails.text = "models/fusion_model_fp32.onnx (20.04 MB) session ready"
        } else {
            ivCheckEngineStatus.setImageResource(R.drawable.ic_speed)
            ivCheckEngineStatus.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_pending))
            tvCheckEngineDetails.text = "Native ONNX session initializing in background…"
        }

        if (isAuthentic && isEngineReady) {
            btnProceedToExecution.isEnabled = true
            btnProceedToExecution.text = "Configure & Run Inference"
        } else if (!isAuthentic) {
            btnProceedToExecution.isEnabled = false
            btnProceedToExecution.text = "Authentic Inputs Required"
        } else {
            btnProceedToExecution.isEnabled = false
            btnProceedToExecution.text = "Initializing Engine…"
        }

        btnProceedToExecution.setOnClickListener {
            host.navigateToInferenceExecution(currentCase.id)
        }

        btnEditSpecimenInputs.setOnClickListener {
            host.navigateToNewSpecimen(currentCase.id)
        }
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"

        fun newInstance(caseId: Long): SpecimenReviewFragment {
            return SpecimenReviewFragment().apply {
                arguments = Bundle().apply {
                    putLong(ARG_CASE_ID, caseId)
                }
            }
        }
    }
}
