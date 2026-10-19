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

    companion object {
        const val PROTOCOL_1_MAX_PATCHES = 1000
    }
}
