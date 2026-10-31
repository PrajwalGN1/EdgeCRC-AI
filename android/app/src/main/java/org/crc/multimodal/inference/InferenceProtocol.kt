package org.crc.multimodal.inference

/**
 * Authoritative evaluation protocols for the CRC Multimodal Fusion Model.
 *
 * PROTOCOL_1_CAPPED:
 * - Manifest-controlled patch selection with an authoritative 1,000-patch cap.
 * - Enforces 1 <= N <= 1000. Inputs with N > 1000 are strictly rejected.
 *
 * PROTOCOL_2_UNCAPPED:
 * - Uncapped candidate patch evaluation preserving all extracted patches.
 * - Supports arbitrary patch counts (including N = 3,914 for CASE_20) subject
 *   to hardware memory and 32-bit allocation safety limits.
 */
enum class InferenceProtocol {
    PROTOCOL_1_CAPPED,
    PROTOCOL_2_UNCAPPED;

    val displayName: String
        get() = when (this) {
            PROTOCOL_1_CAPPED -> "Protocol 1"
            PROTOCOL_2_UNCAPPED -> "Protocol 2"
        }

    val fullDescription: String
        get() = when (this) {
            PROTOCOL_1_CAPPED -> "Protocol 1 (Capped ≤ 1,000 max patches)"
            PROTOCOL_2_UNCAPPED -> "Protocol 2 (Uncapped candidate patches preserved)"
        }

    companion object {
        const val PROTOCOL_1_MAX_PATCHES = 1000

        fun fromStringOrNull(value: String?): InferenceProtocol? {
            if (value.isNullOrBlank()) return null
            val trimmed = value.trim().uppercase()
            return try {
                valueOf(trimmed)
            } catch (e: IllegalArgumentException) {
                when (trimmed) {
                    "PROTOCOL_1", "PROTOCOL 1", "PROTOCOL1" -> PROTOCOL_1_CAPPED
                    "PROTOCOL_2", "PROTOCOL 2", "PROTOCOL2" -> PROTOCOL_2_UNCAPPED
                    else -> null
                }
            }
        }

        fun formatProtocol(protocol: InferenceProtocol?): String {
            return protocol?.displayName ?: "Protocol unavailable"
        }

        fun formatProtocolDetails(protocol: InferenceProtocol?, patchCount: Int? = null): String {
            return when (protocol) {
                PROTOCOL_1_CAPPED -> {
                    if (patchCount != null) "Protocol 1 Capped ($patchCount ≤ 1,000 max patches)"
                    else "Protocol 1 (Capped ≤ 1,000 max patches)"
                }
                PROTOCOL_2_UNCAPPED -> {
                    if (patchCount != null) "Protocol 2 Uncapped ($patchCount candidate patches preserved)"
                    else "Protocol 2 (Uncapped candidate patches preserved)"
                }
                null -> {
                    if (patchCount != null) "Protocol unavailable ($patchCount patches)"
                    else "Protocol unavailable"
                }
            }
        }
    }
}
