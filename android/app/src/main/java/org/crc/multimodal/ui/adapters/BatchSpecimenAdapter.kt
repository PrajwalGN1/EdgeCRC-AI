package org.crc.multimodal.ui.adapters

import android.content.res.ColorStateList
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.chip.Chip
import org.crc.multimodal.R
import org.crc.multimodal.batch.BatchSpecimenItem
import org.crc.multimodal.batch.BatchSpecimenStatus
import java.util.Locale

class BatchSpecimenAdapter(
    private val onItemClickListener: ((BatchSpecimenItem) -> Unit)? = null
) : RecyclerView.Adapter<BatchSpecimenAdapter.ViewHolder>() {

    enum class FilterOption {
        ALL,
        MSI_ONLY,
        MSS_ONLY,
        EXCLUDED_ONLY,
        ARCHIVED_ONLY
    }

    private var allItems: List<BatchSpecimenItem> = emptyList()
    private var displayedItems: List<BatchSpecimenItem> = emptyList()
    private var currentFilter: FilterOption = FilterOption.ALL
    private var currentSearchQuery: String = ""

    // Multi-Selection State
    var isSelectionMode: Boolean = false
        private set
    val selectedSpecimenIds: MutableSet<String> = mutableSetOf()

    var onSelectionChangedListener: ((selectedCount: Int, selectedIds: Set<String>) -> Unit)? = null

    fun setItems(items: List<BatchSpecimenItem>) {
        this.allItems = items
        // Clean up selections that no longer exist
        val currentIds = items.map { it.specimenId }.toSet()
        selectedSpecimenIds.retainAll(currentIds)
        applyFilter()
    }

    fun setFilter(filter: FilterOption) {
        this.currentFilter = filter
        applyFilter()
    }

    fun setSearchQuery(query: String) {
        this.currentSearchQuery = query.trim()
        applyFilter()
    }

    fun setSelectionMode(enabled: Boolean) {
        if (isSelectionMode != enabled) {
            isSelectionMode = enabled
            if (!enabled) {
                selectedSpecimenIds.clear()
            }
            onSelectionChangedListener?.invoke(selectedSpecimenIds.size, selectedSpecimenIds)
            notifyDataSetChanged()
        }
    }

    fun selectAll() {
        for (item in displayedItems) {
            selectedSpecimenIds.add(item.specimenId)
        }
        onSelectionChangedListener?.invoke(selectedSpecimenIds.size, selectedSpecimenIds)
        notifyDataSetChanged()
    }

    fun clearSelection() {
        selectedSpecimenIds.clear()
        onSelectionChangedListener?.invoke(0, selectedSpecimenIds)
        notifyDataSetChanged()
    }

    fun toggleSelection(specimenId: String) {
        if (selectedSpecimenIds.contains(specimenId)) {
            selectedSpecimenIds.remove(specimenId)
        } else {
            selectedSpecimenIds.add(specimenId)
        }
        onSelectionChangedListener?.invoke(selectedSpecimenIds.size, selectedSpecimenIds)
        notifyDataSetChanged()
    }

    fun getSelectedItems(): List<BatchSpecimenItem> {
        return allItems.filter { selectedSpecimenIds.contains(it.specimenId) }
    }

    private fun applyFilter() {
        val baseFiltered = when (currentFilter) {
            FilterOption.ARCHIVED_ONLY -> allItems.filter { it.isArchived }
            FilterOption.ALL -> allItems.filter { !it.isArchived }
            FilterOption.MSI_ONLY -> allItems.filter { !it.isArchived && it.predictedClass == 1 }
            FilterOption.MSS_ONLY -> allItems.filter { !it.isArchived && it.predictedClass == 0 }
            FilterOption.EXCLUDED_ONLY -> allItems.filter {
                !it.isArchived && (it.status == BatchSpecimenStatus.REJECTED ||
                        it.status == BatchSpecimenStatus.FAILED ||
                        it.status == BatchSpecimenStatus.CANCELLED)
            }
        }

        displayedItems = if (currentSearchQuery.isNotBlank()) {
            baseFiltered.filter { it.specimenId.contains(currentSearchQuery, ignoreCase = true) }
        } else {
            baseFiltered
        }
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_batch_specimen, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(displayedItems[position])
    }

    override fun getItemCount(): Int = displayedItems.size

    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val cbSelect: MaterialCheckBox = itemView.findViewById(R.id.cbItemSelect)
        private val tvSpecimenId: TextView = itemView.findViewById(R.id.tvItemSpecimenId)
        private val chipClass: Chip = itemView.findViewById(R.id.chipItemClass)
        private val chipStatus: Chip = itemView.findViewById(R.id.chipItemStatus)
        private val tvProbability: TextView = itemView.findViewById(R.id.tvItemProbability)
        private val tvMetrics: TextView = itemView.findViewById(R.id.tvItemMetrics)
        private val tvDigest: TextView = itemView.findViewById(R.id.tvItemDigest)
        private val tvProtocol: TextView = itemView.findViewById(R.id.tvItemProtocol)
        private val tvError: TextView = itemView.findViewById(R.id.tvItemError)

        fun bind(item: BatchSpecimenItem) {
            val context = itemView.context
            tvSpecimenId.text = item.specimenId

            // Multi-Selection Checkbox & Touch Target
            if (isSelectionMode) {
                cbSelect.visibility = View.VISIBLE
                cbSelect.isChecked = selectedSpecimenIds.contains(item.specimenId)
                cbSelect.contentDescription = "Select specimen ${item.specimenId}"
                cbSelect.setOnClickListener {
                    toggleSelection(item.specimenId)
                }
                itemView.setOnClickListener {
                    toggleSelection(item.specimenId)
                }
            } else {
                cbSelect.visibility = View.GONE
                cbSelect.isChecked = false
                itemView.setOnClickListener {
                    onItemClickListener?.invoke(item)
                }
            }

            // Digest & Provenance
            val digestShort = if (item.bundleSha256.isNotBlank()) item.bundleSha256.take(12) + "…" else "N/A"
            tvDigest.text = "Digest: $digestShort · ${item.provenance.name}"

            // Metrics & Protocol
            val patchesStr = if (item.patchCount > 0) "${item.patchCount} patches" else "No patches"
            val latencyStr = if (item.inferenceLatencyMs > 0) " · ${item.inferenceLatencyMs}ms" else ""
            val schemaStr = if (item.schemaVersion > 0) " · v${item.schemaVersion}" else ""
            val protocolDisplay = org.crc.multimodal.inference.InferenceProtocol.fromStringOrNull(item.protocol)?.displayName ?: "Protocol 1"
            tvMetrics.text = "$patchesStr · $protocolDisplay$latencyStr$schemaStr"

            // Protocol Note / Fallback Reason
            if (item.protocol == org.crc.multimodal.inference.InferenceProtocol.PROTOCOL_2_UNCAPPED.name || !item.protocolReason.isNullOrBlank()) {
                tvProtocol.visibility = View.VISIBLE
                tvProtocol.text = item.protocolReason ?: "Evaluation Mode: Protocol 2 (Uncapped candidate patches)"
            } else {
                tvProtocol.visibility = View.GONE
            }

            // Results and Badges
            if (item.isArchived) {
                chipClass.visibility = View.GONE
                chipStatus.visibility = View.VISIBLE
                val archLabel = if (item.predictedLabel != null) "ARCHIVED (${item.predictedLabel})" else "ARCHIVED"
                chipStatus.text = archLabel
                chipStatus.chipBackgroundColor = ColorStateList.valueOf(ContextCompat.getColor(context, R.color.app_surface_variant))
                chipStatus.setTextColor(ContextCompat.getColor(context, R.color.text_secondary))

                if (item.probability != null) {
                    tvProbability.visibility = View.VISIBLE
                    val probFormatted = String.format(Locale.US, "%.5f", item.probability)
                    val logitFormatted = String.format(Locale.US, "%.4f", item.logit ?: 0f)
                    tvProbability.text = "MSI Probability: $probFormatted | Logit: $logitFormatted"
                } else {
                    tvProbability.visibility = View.GONE
                }
                tvError.visibility = View.GONE
            } else if (item.status == BatchSpecimenStatus.COMPLETED && item.probability != null) {
                chipStatus.visibility = View.GONE
                chipClass.visibility = View.VISIBLE

                val isMsi = item.predictedClass == 1
                val label = if (isMsi) "MSI (P ≥ 0.83)" else "MSS (P < 0.83)"
                chipClass.text = label

                val chipBg = org.crc.multimodal.batch.BatchStatusPalette.resolveColorRes(item.status, item.predictedClass)
                val chipTextColor = org.crc.multimodal.batch.BatchStatusPalette.resolveTextColorRes(item.status, item.predictedClass)
                chipClass.chipBackgroundColor = ColorStateList.valueOf(ContextCompat.getColor(context, chipBg))
                chipClass.setTextColor(ContextCompat.getColor(context, chipTextColor))

                tvProbability.visibility = View.VISIBLE
                val probFormatted = String.format(Locale.US, "%.5f", item.probability)
                val logitFormatted = String.format(Locale.US, "%.4f", item.logit ?: 0f)
                tvProbability.text = "MSI Probability: $probFormatted | Logit: $logitFormatted"

                tvError.visibility = View.GONE
            } else {
                chipClass.visibility = View.GONE
                chipStatus.visibility = View.VISIBLE
                chipStatus.text = item.status.name

                val statusBg = org.crc.multimodal.batch.BatchStatusPalette.resolveColorRes(item.status, item.predictedClass)
                val statusTextColor = org.crc.multimodal.batch.BatchStatusPalette.resolveTextColorRes(item.status, item.predictedClass)
                chipStatus.chipBackgroundColor = ColorStateList.valueOf(ContextCompat.getColor(context, statusBg))
                chipStatus.setTextColor(ContextCompat.getColor(context, statusTextColor))

                tvProbability.visibility = View.GONE

                val errorMsg = item.validationError ?: item.errorDetails
                if (errorMsg != null) {
                    tvError.visibility = View.VISIBLE
                    val prefix = if (item.status == BatchSpecimenStatus.REJECTED) "Exclusion Policy: " else "Execution Failure: "
                    tvError.text = "$prefix$errorMsg"
                    val errorColor = if (item.status == BatchSpecimenStatus.REJECTED) R.color.batch_status_rejected else R.color.batch_status_failed
                    tvError.setTextColor(ContextCompat.getColor(context, errorColor))
                } else {
                    tvError.visibility = View.GONE
                }
            }
        }
    }
}
