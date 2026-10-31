package org.crc.multimodal

import org.crc.multimodal.analytics.SpecimenPredictionRecord
import org.crc.multimodal.data.InMemoryWorkstationRepository
import org.crc.multimodal.evaluation.CohortEvaluationEngine
import org.crc.multimodal.evaluation.EvaluationFilterStatus
import org.crc.multimodal.evaluation.EvaluationOutcome
import org.crc.multimodal.evaluation.EvaluationSortOption
import org.crc.multimodal.evaluation.MetricValue
import org.crc.multimodal.evaluation.ReferenceLabel
import org.crc.multimodal.evaluation.ReferenceLabelImporter
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import kotlin.math.abs

class CohortEvaluationTest {

    private lateinit var repository: InMemoryWorkstationRepository

    @Before
    fun setUp() {
        repository = InMemoryWorkstationRepository()
    }

    /**
     * Test 1: Exact numerical parity with scikit-learn on Protocol 1 (Capped, seed=42).
     */
    @Test
    fun testExactParityWithScikitLearnProtocol1() {
        val referenceJsonFile = findReferenceMetricsFile()
        val jsonText = referenceJsonFile.readText()
        val root = JSONObject(jsonText)
        val proto1 = root.getJSONObject("protocol_1")
        val specimensArray = proto1.getJSONArray("specimens")

        val refLabels = mutableListOf<ReferenceLabel>()
        val predictions = mutableListOf<SpecimenPredictionRecord>()

        for (i in 0 until specimensArray.length()) {
            val s = specimensArray.getJSONObject(i)
            val pid = s.getString("specimen_id")
            val gtLabel = s.getInt("ground_truth_label")
            val gtClass = s.getString("ground_truth_class")
            val prob = s.getDouble("predicted_probability")
            val predClass = s.getInt("predicted_class")
            val predLabel = s.getString("predicted_label")

            refLabels.add(
                ReferenceLabel(
                    specimenId = pid,
                    groundTruthLabel = gtLabel,
                    groundTruthClass = gtClass,
                    cohortId = "TCGA_HOLDOUT_66",
                    source = "Gold Reference"
                )
            )

            predictions.add(
                SpecimenPredictionRecord(
                    specimenId = pid,
                    origin = "SINGLE_CASE",
                    status = "SUCCESS",
                    probability = prob.toFloat(),
                    predictedClass = predClass,
                    predictedLabel = predLabel,
                    threshold = 0.83f,
                    signedMargin = (prob - 0.83).toFloat(),
                    absMargin = abs(prob - 0.83).toFloat(),
                    patchCount = 1000,
                    latencyMs = 25.0,
                    executionTimestamp = 1000L + i
                )
            )
        }

        val summary = CohortEvaluationEngine.evaluateCohort(refLabels, predictions, "TCGA_HOLDOUT_66")

        assertEquals(66, summary.matchedCount)
        assertEquals(0, summary.unmatchedReferenceCount)
        assertEquals(0, summary.unmatchedPredictionCount)

        // Confusion Matrix: TP=12, TN=47, FP=3, FN=4
        val cm = summary.confusionMatrix
        assertEquals(12, cm.tp)
        assertEquals(47, cm.tn)
        assertEquals(3, cm.fp)
        assertEquals(4, cm.fn)
        assertEquals(16, cm.actualPositive)
        assertEquals(50, cm.actualNegative)
        assertEquals(15, cm.predictedPositive)
        assertEquals(51, cm.predictedNegative)
        assertEquals(66, cm.total)

        // Scalar Diagnostic Metrics
        val expectedAcc = 59.0 / 66.0
        val expectedPrec = 12.0 / 15.0
        val expectedRec = 12.0 / 16.0
        val expectedSpec = 47.0 / 50.0
        val expectedBacc = (expectedRec + expectedSpec) / 2.0
        val expectedNpv = 47.0 / 51.0
        val expectedF1 = 24.0 / 31.0
        val expectedRoc = 0.92625000
        val expectedPr = 0.82204595

        assertMetricEquals(expectedAcc, summary.accuracy, 1e-6)
        assertMetricEquals(expectedPrec, summary.precision, 1e-6)
        assertMetricEquals(expectedRec, summary.recallSensitivity, 1e-6)
        assertMetricEquals(expectedSpec, summary.specificity, 1e-6)
        assertMetricEquals(expectedBacc, summary.balancedAccuracy, 1e-6)
        assertMetricEquals(expectedNpv, summary.npv, 1e-6)
        assertMetricEquals(expectedF1, summary.f1Score, 1e-6)
        assertMetricEquals(expectedRoc, summary.rocAuc, 1e-6)
        assertMetricEquals(expectedPr, summary.prAuc, 1e-6)
    }

    /**
     * Test 2: Exact numerical parity with scikit-learn on Protocol 2 (Uncapped candidate).
     */
    @Test
    fun testExactParityWithScikitLearnProtocol2() {
        val referenceJsonFile = findReferenceMetricsFile()
        val jsonText = referenceJsonFile.readText()
        val root = JSONObject(jsonText)
        val proto2 = root.getJSONObject("protocol_2")
        val specimensArray = proto2.getJSONArray("specimens")

        val refLabels = mutableListOf<ReferenceLabel>()
        val predictions = mutableListOf<SpecimenPredictionRecord>()

        for (i in 0 until specimensArray.length()) {
            val s = specimensArray.getJSONObject(i)
            val pid = s.getString("specimen_id")
            val gtLabel = s.getInt("ground_truth_label")
            val gtClass = s.getString("ground_truth_class")
            val prob = s.getDouble("predicted_probability")
            val predClass = s.getInt("predicted_class")
            val predLabel = s.getString("predicted_label")

            refLabels.add(
                ReferenceLabel(
                    specimenId = pid,
                    groundTruthLabel = gtLabel,
                    groundTruthClass = gtClass,
                    cohortId = "TCGA_HOLDOUT_66",
                    source = "Gold Reference"
                )
            )

            predictions.add(
                SpecimenPredictionRecord(
                    specimenId = pid,
                    origin = "SINGLE_CASE",
                    status = "SUCCESS",
                    probability = prob.toFloat(),
                    predictedClass = predClass,
                    predictedLabel = predLabel,
                    threshold = 0.83f,
                    signedMargin = (prob - 0.83).toFloat(),
                    absMargin = abs(prob - 0.83).toFloat(),
                    patchCount = 500,
                    latencyMs = 20.0,
                    executionTimestamp = 2000L + i
                )
            )
        }

        val summary = CohortEvaluationEngine.evaluateCohort(refLabels, predictions, "TCGA_HOLDOUT_66")

        assertEquals(66, summary.matchedCount)

        // Confusion Matrix: TP=12, TN=46, FP=4, FN=4
        val cm = summary.confusionMatrix
        assertEquals(12, cm.tp)
        assertEquals(46, cm.tn)
        assertEquals(4, cm.fp)
        assertEquals(4, cm.fn)

        val expectedAcc = 58.0 / 66.0
        val expectedPrec = 12.0 / 16.0
        val expectedRec = 12.0 / 16.0
        val expectedSpec = 46.0 / 50.0
        val expectedBacc = (expectedRec + expectedSpec) / 2.0
        val expectedNpv = 46.0 / 50.0
        val expectedF1 = 24.0 / 32.0
        val expectedRoc = 0.93375000
        val expectedPr = 0.83683294

        assertMetricEquals(expectedAcc, summary.accuracy, 1e-6)
        assertMetricEquals(expectedPrec, summary.precision, 1e-6)
        assertMetricEquals(expectedRec, summary.recallSensitivity, 1e-6)
        assertMetricEquals(expectedSpec, summary.specificity, 1e-6)
        assertMetricEquals(expectedBacc, summary.balancedAccuracy, 1e-6)
        assertMetricEquals(expectedNpv, summary.npv, 1e-6)
        assertMetricEquals(expectedF1, summary.f1Score, 1e-6)
        assertMetricEquals(expectedRoc, summary.rocAuc, 1e-6)
        assertMetricEquals(expectedPr, summary.prAuc, 1e-6)
    }

    /**
     * Test 3: Handling of undefined metrics due to cohort composition.
     */
    @Test
    fun testUndefinedMetricsHandling() {
        // Case A: Empty dataset
        val emptySummary = CohortEvaluationEngine.evaluateCohort(emptyList(), emptyList())
        assertEquals(0, emptySummary.matchedCount)
        assertTrue(emptySummary.accuracy is MetricValue.Undefined)
        assertTrue(emptySummary.precision is MetricValue.Undefined)
        assertTrue(emptySummary.recallSensitivity is MetricValue.Undefined)
        assertTrue(emptySummary.specificity is MetricValue.Undefined)
        assertTrue(emptySummary.rocAuc is MetricValue.Undefined)
        assertTrue(emptySummary.prAuc is MetricValue.Undefined)

        // Case B: All MSS (no actual positive cases)
        val allMssLabels = listOf(
            ReferenceLabel("S1", 0, "MSS"),
            ReferenceLabel("S2", 0, "MSS")
        )
        val allMssPreds = listOf(
            createPredictionRecord("S1", 0.10, 0),
            createPredictionRecord("S2", 0.20, 0)
        )
        val mssSummary = CohortEvaluationEngine.evaluateCohort(allMssLabels, allMssPreds)
        assertTrue(mssSummary.recallSensitivity is MetricValue.Undefined)
        assertTrue(mssSummary.rocAuc is MetricValue.Undefined)
        assertTrue(mssSummary.prAuc is MetricValue.Undefined)
        assertTrue(mssSummary.specificity is MetricValue.Defined)
        assertEquals(1.0, (mssSummary.specificity as MetricValue.Defined).value, 1e-6)

        // Case C: All MSI (no actual negative cases)
        val allMsiLabels = listOf(
            ReferenceLabel("S1", 1, "MSI"),
            ReferenceLabel("S2", 1, "MSI")
        )
        val allMsiPreds = listOf(
            createPredictionRecord("S1", 0.95, 1),
            createPredictionRecord("S2", 0.88, 1)
        )
        val msiSummary = CohortEvaluationEngine.evaluateCohort(allMsiLabels, allMsiPreds)
        assertTrue(msiSummary.specificity is MetricValue.Undefined)
        assertTrue(msiSummary.rocAuc is MetricValue.Undefined)
        assertTrue(msiSummary.recallSensitivity is MetricValue.Defined)
        assertEquals(1.0, (msiSummary.recallSensitivity as MetricValue.Defined).value, 1e-6)
    }

    /**
     * Test 4: ReferenceLabelImporter schema parsing and rejection of invalid data.
     */
    @Test
    fun testReferenceLabelImporterValidation() {
        // Valid JSON
        val validJson = """
            {
              "cohort_id": "TEST_COHORT",
              "labels": [
                {"specimen_id": "SPEC_01", "ground_truth_label": 1, "ground_truth_class": "MSI", "source": "Lab A"},
                {"specimen_id": "SPEC_02", "ground_truth_label": 0, "ground_truth_class": "MSS", "source": "Lab B"}
              ]
            }
        """.trimIndent()
        val jsonRes = ReferenceLabelImporter.parseJson(validJson)
        assertTrue(jsonRes.isSuccess)
        assertEquals(2, jsonRes.importedLabels.size)
        assertEquals("TEST_COHORT", jsonRes.cohortId)

        // Valid CSV
        val validCsv = """
            specimen_id,ground_truth_label,ground_truth_class,source
            SPEC_01,1,MSI,Lab A
            SPEC_02,0,MSS,Lab B
        """.trimIndent()
        val csvRes = ReferenceLabelImporter.parseCsv(validCsv, "CSV_COHORT")
        assertTrue(csvRes.isSuccess)
        assertEquals(2, csvRes.importedLabels.size)

        // Invalid JSON: label out of bounds
        val invalidLabelJson = """
            {
              "cohort_id": "TEST",
              "labels": [
                {"specimen_id": "SPEC_BAD", "ground_truth_label": 2, "ground_truth_class": "MSI"}
              ]
            }
        """.trimIndent()
        val invalidRes = ReferenceLabelImporter.parseJson(invalidLabelJson)
        assertFalse(invalidRes.isSuccess)
        assertTrue(invalidRes.errorMessages.isNotEmpty())

        // Invalid CSV: mismatched label and class
        val mismatchCsv = """
            specimen_id,ground_truth_label,ground_truth_class,source
            SPEC_01,1,MSS,Conflict
        """.trimIndent()
        val mismatchRes = ReferenceLabelImporter.parseCsv(mismatchCsv)
        assertFalse(mismatchRes.isSuccess)
    }

    /**
     * Test 5: SQLite transactional persistence and complete separation of reference labels.
     */
    @Test
    fun testSqliteReferenceLabelPersistenceAndSeparation() {
        val label1 = ReferenceLabel("SPEC_A", 1, "MSI", "COHORT_1", "Source A")
        val label2 = ReferenceLabel("SPEC_B", 0, "MSS", "COHORT_1", "Source B")
        val label3 = ReferenceLabel("SPEC_C", 0, "MSS", "COHORT_2", "Source C")

        val count = repository.insertReferenceLabels(listOf(label1, label2, label3))
        assertEquals(3, count)

        val retrievedA = repository.getReferenceLabel("SPEC_A")
        assertNotNull(retrievedA)
        assertEquals(1, retrievedA!!.groundTruthLabel)
        assertEquals("MSI", retrievedA.groundTruthClass)

        val allLabels = repository.getAllReferenceLabels()
        assertEquals(3, allLabels.size)

        // Verify independent clearing
        val deleted = repository.deleteReferenceLabel("SPEC_A")
        assertTrue(deleted)
        assertNull(repository.getReferenceLabel("SPEC_A"))
        assertEquals(2, repository.getAllReferenceLabels().size)

        val cleared = repository.clearReferenceLabels()
        assertEquals(2, cleared)
        assertEquals(0, repository.getAllReferenceLabels().size)
    }

    /**
     * Test 6: Specimen matching, cross-specimen isolation, and explorer filtering/sorting.
     */
    @Test
    fun testSpecimenMatchingAndFiltering() {
        val refLabels = listOf(
            ReferenceLabel("CASE_1", 1, "MSI"), // TP
            ReferenceLabel("CASE_2", 0, "MSS"), // TN
            ReferenceLabel("CASE_3", 0, "MSS"), // FP
            ReferenceLabel("CASE_4", 1, "MSI"), // FN
            ReferenceLabel("CASE_UNMATCHED_REF", 0, "MSS") // No prediction
        )

        val preds = listOf(
            createPredictionRecord("CASE_1", 0.95, 1), // TP
            createPredictionRecord("CASE_2", 0.15, 0), // TN
            createPredictionRecord("CASE_3", 0.85, 1), // FP
            createPredictionRecord("CASE_4", 0.50, 0), // FN
            createPredictionRecord("CASE_UNMATCHED_PRED", 0.90, 1) // No reference label
        )

        val summary = CohortEvaluationEngine.evaluateCohort(refLabels, preds, "TEST")
        assertEquals(4, summary.matchedCount)
        assertEquals(1, summary.unmatchedReferenceCount)
        assertEquals(1, summary.unmatchedPredictionCount)

        val matches = summary.specimenMatches
        assertEquals(4, matches.size)

        // Filter tests
        val tpMatches = CohortEvaluationEngine.filterAndSortMatches(matches, EvaluationFilterStatus.TP_ONLY, EvaluationSortOption.SPECIMEN_ID)
        assertEquals(1, tpMatches.size)
        assertEquals("CASE_1", tpMatches[0].specimenId)

        val errorMatches = CohortEvaluationEngine.filterAndSortMatches(matches, EvaluationFilterStatus.MISCLASSIFIED_ONLY, EvaluationSortOption.SPECIMEN_ID)
        assertEquals(2, errorMatches.size) // FP and FN

        // Sort tests
        val sortedMarginAsc = CohortEvaluationEngine.filterAndSortMatches(matches, EvaluationFilterStatus.ALL, EvaluationSortOption.MARGIN_SMALLEST_FIRST)
        assertTrue(sortedMarginAsc[0].decisionMargin <= sortedMarginAsc[1].decisionMargin)
    }

    private fun assertMetricEquals(expected: Double, metric: MetricValue, tolerance: Double) {
        assertTrue("Metric expected to be Defined but was $metric", metric is MetricValue.Defined)
        val actual = (metric as MetricValue.Defined).value
        assertEquals("Metric mismatch: expected $expected, got $actual", expected, actual, tolerance)
    }

    private fun createPredictionRecord(specimenId: String, prob: Double, predClass: Int): SpecimenPredictionRecord {
        return SpecimenPredictionRecord(
            specimenId = specimenId,
            origin = "TEST",
            status = "SUCCESS",
            probability = prob.toFloat(),
            predictedClass = predClass,
            predictedLabel = if (predClass == 1) "MSI" else "MSS",
            threshold = 0.83f,
            signedMargin = (prob - 0.83).toFloat(),
            absMargin = abs(prob - 0.83).toFloat(),
            patchCount = 100,
            latencyMs = 15.0,
            executionTimestamp = System.currentTimeMillis()
        )
    }

    @Test
    fun testCohortEvaluationLayoutXmlValidity() {
        val candidates = listOf(
            File("src/main/res/layout/fragment_cohort_evaluation.xml"),
            File("android/app/src/main/res/layout/fragment_cohort_evaluation.xml"),
            File("../android/app/src/main/res/layout/fragment_cohort_evaluation.xml"),
            File("../../android/app/src/main/res/layout/fragment_cohort_evaluation.xml")
        )
        val layoutFile = candidates.firstOrNull { it.exists() }
            ?: error("Could not find fragment_cohort_evaluation.xml")

        val docBuilder = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder()
        val doc = docBuilder.parse(layoutFile)
        val allNodes = doc.getElementsByTagName("*")

        val missingAttributes = mutableListOf<String>()
        for (i in 0 until allNodes.length) {
            val element = allNodes.item(i) as org.w3c.dom.Element
            val tag = element.tagName
            if (tag == "merge" || tag == "include") continue
            val hasWidth = element.hasAttribute("android:layout_width")
            val hasHeight = element.hasAttribute("android:layout_height")
            val id = element.getAttribute("android:id")
            if (!hasWidth || !hasHeight) {
                missingAttributes.add("<$tag id='$id' hasWidth=$hasWidth hasHeight=$hasHeight>")
            }
        }

        assertTrue(
            "Found XML elements missing layout_width or layout_height in fragment_cohort_evaluation.xml: $missingAttributes",
            missingAttributes.isEmpty()
        )
    }

    private fun findReferenceMetricsFile(): File {
        val candidates = listOf(
            File("../../reports/mobile_deployment/cohort_evaluation_reference_metrics.json"),
            File("../reports/mobile_deployment/cohort_evaluation_reference_metrics.json"),
            File("reports/mobile_deployment/cohort_evaluation_reference_metrics.json"),
            File("cohort_evaluation_reference_metrics.json")
        )
        return candidates.firstOrNull { it.exists() }
            ?: error("Could not find cohort_evaluation_reference_metrics.json")
    }
}

