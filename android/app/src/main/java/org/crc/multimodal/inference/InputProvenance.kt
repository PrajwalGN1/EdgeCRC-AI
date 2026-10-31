package org.crc.multimodal.inference

/**
 * Enumeration specifying the provenance and authenticity of input feature embeddings.
 *
 * Invariants:
 * - [AUTHENTIC_IMPORTED]: Genuine multimodal feature tensors imported from external verified bundles.
 * - [BENCHMARK_REFERENCE]: Frozen authentic reference benchmark cases packaged with the validation holdout.
 * - [DEMONSTRATION_SIMULATED]: Explicitly simulated preview vectors for UI demonstration only.
 *   MUST NEVER be presented as a genuine specimen inference or used for model parity verification.
 */
enum class InputProvenance(val displayName: String, val isAuthentic: Boolean) {
    AUTHENTIC_IMPORTED("Authentic Imported Bundle", true),
    BENCHMARK_REFERENCE("Frozen Benchmark Reference", true),
    DEMONSTRATION_SIMULATED("Simulated Demonstration Preview", false)
}
