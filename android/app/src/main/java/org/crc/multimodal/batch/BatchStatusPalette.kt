package org.crc.multimodal.batch

import org.crc.multimodal.R

/**
 * Healthcare UI status color palette for CRC Batch Specimen Analysis.
 *
 * Strict Color System:
 * - MSS:      #2ECC9A (Emerald Green) - Model Classification Class 0 (P < 0.83)
 * - MSI:      #F56B70 (Coral Red)     - Model Classification Class 1 (P ≥ 0.83)
 * - FAILED:   #F59E0B (Amber Orange)  - Execution / Processing Failure
 * - REJECTED: #D6A83D (Gold)          - Policy Exclusion / Precondition Not Met
 * - TEXT:     #101820 (Dark Slate)    - High contrast text on bright badge chips (WCAG AAA)
 */
object BatchStatusPalette {

    const val COLOR_MSS = "#2ECC9A"
    const val COLOR_MSI = "#F56B70"
    const val COLOR_FAILED = "#F59E0B"
    const val COLOR_REJECTED = "#D6A83D"
    const val COLOR_TEXT_DARK = "#101820"
    const val COLOR_FALLBACK = "#888888"

    /**
     * Resolves hex color code from status name or classification label.
     */
    fun resolveColorHex(status: String?): String {
        return when (status?.trim()?.uppercase()) {
            "MSS" -> COLOR_MSS
            "MSI" -> COLOR_MSI
            "FAILED" -> COLOR_FAILED
            "REJECTED" -> COLOR_REJECTED
            else -> COLOR_FALLBACK
        }
    }

    /**
     * Resolves hex color code from batch specimen status and predicted class.
     */
    fun resolveColorHex(status: BatchSpecimenStatus, predictedClass: Int? = null): String {
        return when (status) {
            BatchSpecimenStatus.COMPLETED -> if (predictedClass == 1) COLOR_MSI else COLOR_MSS
            BatchSpecimenStatus.FAILED -> COLOR_FAILED
            BatchSpecimenStatus.REJECTED -> COLOR_REJECTED
            else -> COLOR_FALLBACK
        }
    }

    /**
     * Resolves background color resource ID for status chip rendering.
     */
    fun resolveColorRes(status: BatchSpecimenStatus, predictedClass: Int? = null): Int {
        return when (status) {
            BatchSpecimenStatus.COMPLETED -> if (predictedClass == 1) R.color.batch_status_msi else R.color.batch_status_mss
            BatchSpecimenStatus.FAILED -> R.color.batch_status_failed
            BatchSpecimenStatus.REJECTED -> R.color.batch_status_rejected
            BatchSpecimenStatus.VALIDATED -> R.color.primary_dark
            BatchSpecimenStatus.INFERRING -> R.color.status_pending
            BatchSpecimenStatus.CANCELLED -> R.color.text_secondary_light
            else -> R.color.text_secondary_light
        }
    }

    /**
     * Resolves text color resource ID for badge chips to guarantee accessible contrast.
     */
    fun resolveTextColorRes(status: BatchSpecimenStatus, predictedClass: Int? = null): Int {
        return when (status) {
            BatchSpecimenStatus.COMPLETED,
            BatchSpecimenStatus.FAILED,
            BatchSpecimenStatus.REJECTED -> R.color.batch_status_text
            else -> android.R.color.white
        }
    }
}
