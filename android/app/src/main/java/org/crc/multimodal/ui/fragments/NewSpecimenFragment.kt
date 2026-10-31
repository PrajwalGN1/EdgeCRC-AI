package org.crc.multimodal.ui.fragments

import android.content.Context
import android.os.Bundle
import android.view.View
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.button.MaterialButtonToggleGroup
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import org.crc.multimodal.R
import org.crc.multimodal.data.CaseStatus
import org.crc.multimodal.data.ResearchCase
import org.crc.multimodal.inference.InferenceProtocol
import org.crc.multimodal.ui.WorkstationNavigationHost
import kotlin.random.Random

import android.net.Uri
import android.widget.ImageView
import android.widget.LinearLayout
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import org.crc.multimodal.inference.AuthenticBundleLoader
import org.crc.multimodal.inference.AuthenticFeatureBundle
import org.crc.multimodal.inference.BundleStorageManager
import org.crc.multimodal.inference.NpyTensorLoader
import org.crc.multimodal.inference.ReferenceTensorLoader
import org.crc.multimodal.data.SpecimenInput
import java.io.InputStream

class NewSpecimenFragment : Fragment(R.layout.fragment_new_specimen) {

    private var navHost: WorkstationNavigationHost? = null
    private var selectedPatchCount: Int = 10
    private var editingCaseId: Long? = null
    private var loadedBundle: AuthenticFeatureBundle? = null
    private var currentImportFilename: String = ""

    // UI elements for import status and metadata
    private var layoutImportStatus: LinearLayout? = null
    private var ivImportStatusIcon: ImageView? = null
    private var tvImportStatusDetails: TextView? = null

    private var layoutIngestionDetails: LinearLayout? = null
    private var tvValidationStatusBadge: TextView? = null
    private var tvSpecimenIdDisplay: TextView? = null
    private var tvSelectedFilename: TextView? = null
    private var tvVisionShapeDisplay: TextView? = null
    private var tvTextShapeDisplay: TextView? = null
    private var tvErrorExplanation: TextView? = null
    private var layoutIngestionActions: LinearLayout? = null
    private var btnConfirmImport: MaterialButton? = null
    private var btnStartInference: MaterialButton? = null

    private var etCaseIdRef: TextInputEditText? = null
    private var etSpecimenDescRef: TextInputEditText? = null
    private var etResearchContextRef: TextInputEditText? = null
    private var tvProtocolBadgeRef: TextView? = null
    private var tvTensorDimRef: TextView? = null
    private var tvDetectedPatchCountRef: TextView? = null
    private var tvPatchCountSourceRef: TextView? = null

    // Launchers
    private val openBundleLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        uri?.let { handleImportedBundleUri(it) }
    }

    private var pendingVisionUri: Uri? = null
    private var pendingVisionFilename: String? = null

    private val openVisionPickerLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            pendingVisionUri = uri
            pendingVisionFilename = getFileNameFromUri(uri)
            Toast.makeText(requireContext(), "Vision tensor selected: $pendingVisionFilename. Now select Text tensor.", Toast.LENGTH_SHORT).show()
            openTextPickerLauncher.launch(arrayOf("*/*"))
        }
    }

    private val openTextPickerLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        val vUri = pendingVisionUri
        val vName = pendingVisionFilename
        if (uri != null && vUri != null && vName != null) {
            val tName = getFileNameFromUri(uri)
            handleSeparateTensorImport(vUri, vName, uri, tName)
        }
    }

    override fun onAttach(context: Context) {
        super.onAttach(context)
        if (context is WorkstationNavigationHost) {
            navHost = context
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        editingCaseId = arguments?.getLong(ARG_CASE_ID)?.takeIf { it > 0 }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        navHost?.setTopToolbar(
            title = "New Specimen Analysis",
            subtitle = "Multimodal Case Ingestion",
            showBackButton = true
        )

        val host = navHost ?: return
        val repository = host.getRepository()

        val tilCaseId: TextInputLayout = view.findViewById(R.id.tilCaseId)
        val etCaseId: TextInputEditText = view.findViewById(R.id.etCaseId)
        val btnAutoGenerateId: MaterialButton = view.findViewById(R.id.btnAutoGenerateId)
        val tilSpecimenDescription: TextInputLayout = view.findViewById(R.id.tilSpecimenDescription)
        val etSpecimenDescription: TextInputEditText = view.findViewById(R.id.etSpecimenDescription)
        val tilResearchContext: TextInputLayout = view.findViewById(R.id.tilResearchContext)
        val etResearchContext: TextInputEditText = view.findViewById(R.id.etResearchContext)

        val tvDetectedPatchCount: TextView = view.findViewById(R.id.tvDetectedPatchCount)
        val tvPatchCountSource: TextView = view.findViewById(R.id.tvPatchCountSource)
        val tvNewSpecimenProtocolBadge: TextView = view.findViewById(R.id.tvNewSpecimenProtocolBadge)
        val tvNewSpecimenTensorDim: TextView = view.findViewById(R.id.tvNewSpecimenTensorDim)

        val btnSelectBundleFile: MaterialButton = view.findViewById(R.id.btnSelectBundleFile)
        val btnSelectSeparatePair: MaterialButton = view.findViewById(R.id.btnSelectSeparatePair)

        layoutImportStatus = view.findViewById(R.id.layoutImportStatus)
        ivImportStatusIcon = view.findViewById(R.id.ivImportStatusIcon)
        tvImportStatusDetails = view.findViewById(R.id.tvImportStatusDetails)

        layoutIngestionDetails = view.findViewById(R.id.layoutIngestionDetails)
        tvValidationStatusBadge = view.findViewById(R.id.tvValidationStatusBadge)
        tvSpecimenIdDisplay = view.findViewById(R.id.tvSpecimenIdDisplay)
        tvSelectedFilename = view.findViewById(R.id.tvSelectedFilename)
        tvVisionShapeDisplay = view.findViewById(R.id.tvVisionShapeDisplay)
        tvTextShapeDisplay = view.findViewById(R.id.tvTextShapeDisplay)
        tvErrorExplanation = view.findViewById(R.id.tvErrorExplanation)
        layoutIngestionActions = view.findViewById(R.id.layoutIngestionActions)
        btnConfirmImport = view.findViewById(R.id.btnConfirmImport)
        btnStartInference = view.findViewById(R.id.btnStartInference)

        etCaseIdRef = etCaseId
        etSpecimenDescRef = etSpecimenDescription
        etResearchContextRef = etResearchContext
        tvProtocolBadgeRef = tvNewSpecimenProtocolBadge
        tvTensorDimRef = tvNewSpecimenTensorDim
        tvDetectedPatchCountRef = tvDetectedPatchCount
        tvPatchCountSourceRef = tvPatchCountSource

        val btnContinueToReview: MaterialButton = view.findViewById(R.id.btnContinueToReview)
        val btnSaveDraft: MaterialButton = view.findViewById(R.id.btnSaveDraft)
        val btnCancelNewSpecimen: MaterialButton = view.findViewById(R.id.btnCancelNewSpecimen)

        btnSelectBundleFile.setOnClickListener {
            openBundleLauncher.launch(arrayOf("*/*"))
        }

        btnSelectSeparatePair.setOnClickListener {
            Toast.makeText(requireContext(), "Step 1: Select Vision Embedding (.npy)", Toast.LENGTH_SHORT).show()
            openVisionPickerLauncher.launch(arrayOf("*/*"))
        }

        btnConfirmImport?.setOnClickListener {
            confirmImportAndSave {
                tvValidationStatusBadge?.text = "COMMITTED"
            }
        }

        btnStartInference?.setOnClickListener {
            val caseIdText = etCaseId.text?.toString()?.trim().orEmpty()
            val description = etSpecimenDescription.text?.toString()?.trim().orEmpty()
            val contextText = etResearchContext.text?.toString()?.trim().orEmpty()

            var isValid = true
            if (caseIdText.isBlank()) {
                tilCaseId.error = "Case ID is required"
                isValid = false
            } else {
                tilCaseId.error = null
            }

            if (description.isBlank()) {
                tilSpecimenDescription.error = "Specimen description is required"
                isValid = false
            } else {
                tilSpecimenDescription.error = null
            }

            if (contextText.length > 50000) {
                tilResearchContext.error = "Clinical context exceeds maximum limit of 50,000 characters (${contextText.length} entered)"
                isValid = false
            } else {
                tilResearchContext.error = null
            }

            if (!isValid) return@setOnClickListener

            val hasBundle = loadedBundle != null || BundleStorageManager.hasBundle(requireContext(), caseIdText)
            val isBenchmark = ReferenceTensorLoader.isReferenceCase(caseIdText)
            if (!hasBundle && !isBenchmark) {
                Toast.makeText(requireContext(), "Authentic feature embeddings are required. Import and confirm a valid .crcbundle before running inference.", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }

            // Prevent duplicate clicks
            btnStartInference?.isEnabled = false

            try {
                if (loadedBundle != null) {
                    confirmImportAndSave { savedId ->
                        btnStartInference?.isEnabled = true
                        host.navigateToInferenceExecution(savedId)
                    }
                } else {
                    val caseToSave = ResearchCase(
                        id = editingCaseId ?: 0L,
                        caseIdentifier = caseIdText,
                        specimenDescription = description,
                        researchContext = contextText,
                        status = CaseStatus.READY_FOR_ANALYSIS,
                        patchCount = selectedPatchCount,
                        protocol = if (selectedPatchCount <= 1000) InferenceProtocol.PROTOCOL_1_CAPPED else InferenceProtocol.PROTOCOL_2_UNCAPPED
                    )
                    val savedId = if (editingCaseId != null && editingCaseId!! > 0) {
                        repository.updateCase(caseToSave)
                        editingCaseId!!
                    } else {
                        repository.insertCase(caseToSave)
                    }
                    btnStartInference?.isEnabled = true
                    host.navigateToInferenceExecution(savedId)
                }
            } catch (e: Exception) {
                btnStartInference?.isEnabled = true
                tilCaseId.error = e.message
            }
        }

        // Patch count display initialization
        selectedPatchCount = 10
        tvDetectedPatchCount.text = "Awaiting specimen tensor"
        tvPatchCountSource.text = "(Determined automatically from authentic bundle)"
        updateTensorSummary(tvNewSpecimenProtocolBadge, tvNewSpecimenTensorDim)

        btnAutoGenerateId.setOnClickListener {
            val autoId = "CRC-CASE-${Random.nextInt(100, 999)}"
            etCaseId.setText(autoId)
            tilCaseId.error = null
        }

        // Prepopulate if editing existing draft
        editingCaseId?.let { id ->
            repository.getCaseById(id)?.let { existingCase ->
                etCaseId.setText(existingCase.caseIdentifier)
                etSpecimenDescription.setText(existingCase.specimenDescription)
                etResearchContext.setText(existingCase.researchContext)
                selectedPatchCount = existingCase.patchCount
                tvDetectedPatchCount.text = "$selectedPatchCount Patches"
                tvPatchCountSource.text = "(Stored case value)"
                updateTensorSummary(tvNewSpecimenProtocolBadge, tvNewSpecimenTensorDim)
            }
        }

        // Save Draft Action
        btnSaveDraft.setOnClickListener {
            val caseIdText = etCaseId.text?.toString()?.trim().orEmpty()
            if (caseIdText.isBlank()) {
                tilCaseId.error = "Please specify a Case ID to save draft"
                return@setOnClickListener
            }
            tilCaseId.error = null

            val description = etSpecimenDescription.text?.toString()?.trim().orEmpty()
            val contextText = etResearchContext.text?.toString()?.trim().orEmpty()

            if (contextText.length > 50000) {
                tilResearchContext.error = "Clinical context exceeds maximum limit of 50,000 characters (${contextText.length} entered)"
                return@setOnClickListener
            }
            tilResearchContext.error = null

            try {
                val caseToSave = ResearchCase(
                    id = editingCaseId ?: 0L,
                    caseIdentifier = caseIdText,
                    specimenDescription = description,
                    researchContext = contextText,
                    status = CaseStatus.DRAFT,
                    patchCount = selectedPatchCount,
                    protocol = if (selectedPatchCount <= 1000) InferenceProtocol.PROTOCOL_1_CAPPED else InferenceProtocol.PROTOCOL_2_UNCAPPED
                )

                if (editingCaseId != null && editingCaseId!! > 0) {
                    repository.updateCase(caseToSave)
                } else {
                    repository.insertCase(caseToSave)
                }

                Toast.makeText(requireContext(), "Draft saved successfully", Toast.LENGTH_SHORT).show()
                host.navigateToCases()
            } catch (e: Exception) {
                tilCaseId.error = e.message
            }
        }

        // Continue to Review Action
        btnContinueToReview.setOnClickListener {
            val caseIdText = etCaseId.text?.toString()?.trim().orEmpty()
            val description = etSpecimenDescription.text?.toString()?.trim().orEmpty()
            val contextText = etResearchContext.text?.toString()?.trim().orEmpty()

            var isValid = true
            if (caseIdText.isBlank()) {
                tilCaseId.error = "Case ID is required"
                isValid = false
            } else {
                tilCaseId.error = null
            }

            if (description.isBlank()) {
                tilSpecimenDescription.error = "Specimen description is required"
                isValid = false
            } else {
                tilSpecimenDescription.error = null
            }

            if (contextText.length > 50000) {
                tilResearchContext.error = "Clinical context exceeds maximum limit of 50,000 characters (${contextText.length} entered)"
                isValid = false
            } else {
                tilResearchContext.error = null
            }

            if (!isValid) return@setOnClickListener

            try {
                val caseToSave = ResearchCase(
                    id = editingCaseId ?: 0L,
                    caseIdentifier = caseIdText,
                    specimenDescription = description,
                    researchContext = contextText,
                    status = CaseStatus.READY_FOR_ANALYSIS,
                    patchCount = selectedPatchCount,
                    protocol = if (selectedPatchCount <= 1000) InferenceProtocol.PROTOCOL_1_CAPPED else InferenceProtocol.PROTOCOL_2_UNCAPPED
                )

                val savedId = if (editingCaseId != null && editingCaseId!! > 0) {
                    repository.updateCase(caseToSave)
                    editingCaseId!!
                } else {
                    repository.insertCase(caseToSave)
                }

                if (loadedBundle != null) {
                    try {
                        repository.insertSpecimenInput(
                            SpecimenInput(
                                caseId = savedId,
                                inputType = "AUTHENTIC_BUNDLE",
                                localUriOrAsset = BundleStorageManager.getBundleFile(requireContext(), caseToSave.caseIdentifier)?.absolutePath ?: "",
                                filename = if (currentImportFilename.isNotBlank()) currentImportFilename else "${caseToSave.caseIdentifier}.crcbundle",
                                validationStatus = "VERIFIED_AUTHENTIC"
                            )
                        )
                    } catch (ignored: Exception) {}
                }

                host.navigateToSpecimenReview(savedId)
            } catch (e: Exception) {
                tilCaseId.error = e.message
            }
        }

        // Cancel with confirmation if fields are non-empty
        btnCancelNewSpecimen.setOnClickListener {
            val hasContent = etCaseId.text?.isNotBlank() == true ||
                    etSpecimenDescription.text?.isNotBlank() == true ||
                    etResearchContext.text?.isNotBlank() == true

            if (hasContent) {
                MaterialAlertDialogBuilder(requireContext())
                    .setTitle("Discard Changes?")
                    .setMessage("You have unsaved specimen data. Discard and leave?")
                    .setPositiveButton("Discard") { _, _ -> host.navigateToWorkspace() }
                    .setNegativeButton("Keep Editing", null)
                    .show()
            } else {
                host.navigateToWorkspace()
            }
        }
    }

    private fun getFileNameFromUri(uri: Uri): String {
        var name = uri.lastPathSegment ?: "unknown_file"
        if (uri.scheme == "content") {
            try {
                requireContext().contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                    val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                    if (nameIndex != -1 && cursor.moveToFirst()) {
                        name = cursor.getString(nameIndex)
                    }
                }
            } catch (_: Exception) {}
        }
        return name
    }

    private fun handleImportedBundleUri(uri: Uri) {
        val filename = getFileNameFromUri(uri)
        try {
            val stream: InputStream = requireContext().contentResolver.openInputStream(uri)
                ?: throw IllegalArgumentException("Cannot open stream from selected URI: $uri")
            val bundle = stream.use { AuthenticBundleLoader.loadFromStream(it) }

            displayImportSuccess(bundle, filename)
        } catch (t: Throwable) {
            displayImportError(filename, t.localizedMessage ?: "Invalid or corrupt bundle file.")
        }
    }

    private fun handleSeparateTensorImport(
        visionUri: Uri,
        visionFilename: String,
        textUri: Uri,
        textFilename: String
    ) {
        val combinedName = "$visionFilename + $textFilename"
        try {
            val visionBytes = requireContext().contentResolver.openInputStream(visionUri)?.use { it.readBytes() }
                ?: throw IllegalArgumentException("Cannot read vision tensor from $visionFilename")
            val textBytes = requireContext().contentResolver.openInputStream(textUri)?.use { it.readBytes() }
                ?: throw IllegalArgumentException("Cannot read text tensor from $textFilename")

            val visionNpy = NpyTensorLoader.loadFromBytes(visionBytes)
            val textNpy = NpyTensorLoader.loadFromBytes(textBytes)

            // Derive specimen ID: e.g. TCGA-A6-3807_vision.npy -> TCGA-A6-3807
            var derivedId = visionFilename.replace("_vision.npy", "", ignoreCase = true)
                .replace(".vision.npy", "", ignoreCase = true)
                .replace(".npy", "", ignoreCase = true)
                .trim()
            if (derivedId.isBlank() || derivedId.equals("vision", ignoreCase = true)) {
                derivedId = etCaseIdRef?.text?.toString()?.trim()?.takeIf { it.isNotBlank() }
                    ?: "CRC-CASE-${Random.nextInt(100, 999)}"
            }

            val bundle = AuthenticBundleLoader.createBundleFromNpyPair(
                specimenId = derivedId,
                description = etSpecimenDescRef?.text?.toString()?.trim().orEmpty(),
                visionNpy = visionNpy,
                textNpy = textNpy
            )

            displayImportSuccess(bundle, combinedName)
        } catch (t: Throwable) {
            displayImportError(combinedName, t.localizedMessage ?: "Failed to validate separate tensor pair.")
        }
    }

    private fun displayImportSuccess(bundle: AuthenticFeatureBundle, filename: String) {
        loadedBundle = bundle
        currentImportFilename = filename

        layoutIngestionDetails?.visibility = View.VISIBLE
        tvValidationStatusBadge?.text = "VALIDATED"
        tvValidationStatusBadge?.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_success_bg)
        tvValidationStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_success))

        tvSpecimenIdDisplay?.text = "ID: ${bundle.specimenId}"
        tvSelectedFilename?.text = "File(s): $filename"
        tvVisionShapeDisplay?.text = "Vision: [1, ${bundle.patchCount}, ${bundle.visionDim}] · ${bundle.patchCount} patches"
        tvTextShapeDisplay?.text = "Clinical Text: [1, ${bundle.textDim}] · ClinicalBERT"
        tvErrorExplanation?.visibility = View.GONE
        layoutIngestionActions?.visibility = View.VISIBLE

        // Prepopulate Case ID & Description
        etCaseIdRef?.setText(bundle.specimenId)
        if (etSpecimenDescRef?.text.isNullOrBlank() && bundle.description.isNotBlank()) {
            etSpecimenDescRef?.setText(bundle.description)
        }

        // Configure Patch Count
        selectedPatchCount = bundle.patchCount
        tvDetectedPatchCountRef?.text = "${bundle.patchCount} Patches"
        tvPatchCountSourceRef?.text = "(Authentic .crcbundle verified)"
        tvProtocolBadgeRef?.let { badge ->
            tvTensorDimRef?.let { dim ->
                updateTensorSummary(badge, dim)
            }
        }

        // Backward compatibility
        layoutImportStatus?.visibility = View.VISIBLE
        layoutImportStatus?.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_success_bg)
        ivImportStatusIcon?.setImageResource(R.drawable.ic_check_circle)
        ivImportStatusIcon?.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_success))
        tvImportStatusDetails?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_success))
        tvImportStatusDetails?.text = "AUTHENTIC: ${bundle.specimenId} · ${bundle.patchCount} patches · SHA-256 verified"

        Toast.makeText(requireContext(), "Tensor imported & validated for ${bundle.specimenId}", Toast.LENGTH_SHORT).show()
    }

    private fun displayImportError(filename: String, errorMsg: String) {
        loadedBundle = null
        currentImportFilename = filename

        layoutIngestionDetails?.visibility = View.VISIBLE
        tvValidationStatusBadge?.text = "MALFORMED / REJECTED"
        tvValidationStatusBadge?.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_error_bg)
        tvValidationStatusBadge?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_error))

        tvSpecimenIdDisplay?.text = "ID: Unresolved"
        tvSelectedFilename?.text = "File(s): $filename"
        tvVisionShapeDisplay?.text = "Vision: Invalid"
        tvTextShapeDisplay?.text = "Clinical Text: Invalid"
        tvErrorExplanation?.visibility = View.VISIBLE
        tvErrorExplanation?.text = "Validation Error: $errorMsg"
        layoutIngestionActions?.visibility = View.GONE

        // Backward compatibility
        layoutImportStatus?.visibility = View.VISIBLE
        layoutImportStatus?.backgroundTintList = ContextCompat.getColorStateList(requireContext(), R.color.status_error_bg)
        ivImportStatusIcon?.setImageResource(R.drawable.ic_error_outline)
        ivImportStatusIcon?.setColorFilter(ContextCompat.getColor(requireContext(), R.color.status_error))
        tvImportStatusDetails?.setTextColor(ContextCompat.getColor(requireContext(), R.color.status_error))
        tvImportStatusDetails?.text = "Import rejected: $errorMsg"

        Toast.makeText(requireContext(), "Validation failed: $errorMsg", Toast.LENGTH_LONG).show()
    }

    private fun confirmImportAndSave(onSaved: (Long) -> Unit) {
        val bundle = loadedBundle
        if (bundle == null) {
            Toast.makeText(requireContext(), "No validated tensor bundle to confirm.", Toast.LENGTH_SHORT).show()
            return
        }

        val caseIdText = bundle.specimenId
        val descText = etSpecimenDescRef?.text?.toString()?.trim().orEmpty()
        val contextText = etResearchContextRef?.text?.toString()?.trim().orEmpty()

        if (contextText.length > 50000) {
            Toast.makeText(requireContext(), "Clinical context exceeds maximum limit of 50,000 characters", Toast.LENGTH_LONG).show()
            return
        }

        val host = navHost ?: return
        val repository = host.getRepository()

        // Check if bundle already exists in storage and save
        BundleStorageManager.saveBundle(requireContext(), bundle, overwrite = true)

        val protocol = if (bundle.patchCount <= 1000) {
            InferenceProtocol.PROTOCOL_1_CAPPED
        } else {
            InferenceProtocol.PROTOCOL_2_UNCAPPED
        }

        val caseToSave = ResearchCase(
            id = editingCaseId ?: 0L,
            caseIdentifier = caseIdText,
            specimenDescription = if (descText.isNotBlank()) descText else bundle.description,
            researchContext = contextText,
            status = CaseStatus.READY_FOR_ANALYSIS,
            patchCount = bundle.patchCount,
            protocol = protocol
        )

        val savedId = if (editingCaseId != null && editingCaseId!! > 0) {
            repository.updateCase(caseToSave)
            editingCaseId!!
        } else {
            repository.insertCase(caseToSave)
        }
        editingCaseId = savedId

        try {
            repository.insertSpecimenInput(
                SpecimenInput(
                    caseId = savedId,
                    inputType = "AUTHENTIC_BUNDLE",
                    localUriOrAsset = BundleStorageManager.getBundleFile(requireContext(), caseToSave.caseIdentifier)?.absolutePath ?: "",
                    filename = if (currentImportFilename.isNotBlank()) currentImportFilename else "${caseToSave.caseIdentifier}.crcbundle",
                    validationStatus = "VERIFIED_AUTHENTIC"
                )
            )
        } catch (_: Exception) {}

        Toast.makeText(requireContext(), "Specimen $caseIdText confirmed & ready for inference", Toast.LENGTH_SHORT).show()
        onSaved(savedId)
    }

    private fun updateTensorSummary(tvProtocol: TextView, tvDim: TextView) {
        val protocolText = if (selectedPatchCount <= 1000) {
            "Protocol 1 (Capped ≤ 1,000)"
        } else {
            "Protocol 2 (Uncapped)"
        }
        tvProtocol.text = protocolText
        tvDim.text = "Shape: [1, $selectedPatchCount, 1024]"
    }

    companion object {
        private const val ARG_CASE_ID = "arg_case_id"

        fun newInstance(caseId: Long? = null): NewSpecimenFragment {
            return NewSpecimenFragment().apply {
                arguments = Bundle().apply {
                    if (caseId != null) putLong(ARG_CASE_ID, caseId)
                }
            }
        }
    }
}
