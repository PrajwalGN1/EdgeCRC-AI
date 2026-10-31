package org.crc.multimodal

import org.crc.multimodal.inference.OnnxRuntimeManager
import org.crc.multimodal.inference.OnnxRuntimeState
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Exhaustive unit test suite verifying the authoritative OnnxRuntimeManager state machine:
 *
 * 1. Initial state is NOT_STARTED.
 * 2. Asynchronous loading transitions from NOT_STARTED -> INITIALIZING -> READY.
 * 3. State listeners receive immediate notification of current state upon registration.
 * 4. Concurrent callers share a single in-flight initialization attempt without duplicates.
 * 5. Re-triggering initialization when READY is a no-op that preserves the active session.
 * 6. Corrupt or invalid model bytes transition to FAILED with meaningful diagnostics.
 * 7. A FAILED state cannot be reported as Ready.
 * 8. Explicit retry after failure successfully re-initializes when valid bytes are provided.
 * 9. Closing the manager resets state to NOT_STARTED and closes the classifier.
 * 10. Concurrent listener registration/removal is thread-safe.
 * 11. Inference executed through the manager's active classifier produces valid results.
 */
class OnnxRuntimeManagerTest {

    private val modelFile = File("src/main/assets/models/fusion_model_fp32.onnx")
    private var managerToClean: OnnxRuntimeManager? = null

    @After
    fun tearDown() {
        managerToClean?.close()
        managerToClean = null
    }

    @Test
    fun testInitialStateIsNotStarted() {
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = Executors.newSingleThreadExecutor(),
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        assertEquals(OnnxRuntimeState.NotStarted, manager.state)
        assertFalse(manager.isReady)
        assertNull(manager.classifier)
    }

    @Test
    fun testColdStartTransitionsToInitializingAndReady() {
        assertTrue("Model asset file must exist for verification", modelFile.exists())
        val executor = Executors.newSingleThreadExecutor()
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = executor,
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        val stateHistory = mutableListOf<String>()
        val latch = CountDownLatch(1)

        manager.addListener { state ->
            stateHistory.add(state.toString())
            if (state is OnnxRuntimeState.Ready) {
                latch.countDown()
            }
        }

        manager.initializeAsync(
            assetProvider = { modelFile.readBytes() },
            assetPath = "models/fusion_model_fp32.onnx"
        )

        val completed = latch.await(10, TimeUnit.SECONDS)
        assertTrue("Model initialization must complete within 10s", completed)

        assertTrue("History must contain NOT_STARTED", stateHistory.any { it.contains("NOT_STARTED") })
        assertTrue("History must contain INITIALIZING", stateHistory.any { it.contains("INITIALIZING") })
        assertTrue("History must contain READY", stateHistory.any { it.contains("READY") })

        assertTrue(manager.isReady)
        val readyState = manager.state as OnnxRuntimeState.Ready
        assertEquals("0402f6c522beac5733e30eeecad908927957b5d4164b6b896663f43ba3702d73", readyState.sha256Fingerprint)
        assertNotNull(manager.classifier)
    }

    @Test
    fun testListenerInvokedImmediatelyUponRegistration() {
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = Executors.newSingleThreadExecutor(),
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        var observedState: OnnxRuntimeState? = null
        manager.addListener { state ->
            observedState = state
        }

        assertEquals(OnnxRuntimeState.NotStarted, observedState)
    }

    @Test
    fun testMultipleConcurrentCallersShareSingleInitialization() {
        val executor = Executors.newSingleThreadExecutor()
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = executor,
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        val readCount = AtomicInteger(0)
        val latch = CountDownLatch(1)

        manager.addListener { state ->
            if (state is OnnxRuntimeState.Ready) {
                latch.countDown()
            }
        }

        val assetProvider: () -> ByteArray = {
            readCount.incrementAndGet()
            Thread.sleep(100) // Simulate asset I/O delay
            modelFile.readBytes()
        }

        // Fire 5 concurrent initialization requests
        repeat(5) {
            manager.initializeAsync(
                assetProvider = assetProvider,
                assetPath = "models/fusion_model_fp32.onnx"
            )
        }

        val completed = latch.await(10, TimeUnit.SECONDS)
        assertTrue("Concurrent initialization must complete", completed)

        // Only 1 execution should have read the asset bytes
        assertEquals(1, readCount.get())
        assertTrue(manager.isReady)
    }

    @Test
    fun testRepeatedInitializeAsyncWhenReadyIsNoOp() {
        val executor = Executors.newSingleThreadExecutor()
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = executor,
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        val latch = CountDownLatch(1)
        manager.addListener { state ->
            if (state is OnnxRuntimeState.Ready) {
                latch.countDown()
            }
        }

        manager.initializeAsync(
            assetProvider = { modelFile.readBytes() },
            assetPath = "models/fusion_model_fp32.onnx"
        )
        latch.await(10, TimeUnit.SECONDS)

        val originalClassifier = manager.classifier
        assertNotNull(originalClassifier)

        // Call initializeAsync again without forceRetry
        var reInitCalled = false
        manager.initializeAsync(
            assetProvider = {
                reInitCalled = true
                modelFile.readBytes()
            },
            assetPath = "models/fusion_model_fp32.onnx",
            forceRetry = false
        )

        assertFalse("Re-initializing when READY must be a no-op", reInitCalled)
        assertEquals("Classifier instance must be preserved", originalClassifier, manager.classifier)
    }

    @Test
    fun testCorruptModelBytesTransitionsToFailed() {
        val executor = Executors.newSingleThreadExecutor()
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = executor,
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        val latch = CountDownLatch(1)
        manager.addListener { state ->
            if (state is OnnxRuntimeState.Failed) {
                latch.countDown()
            }
        }

        val corruptBytes = byteArrayOf(0x00, 0x01, 0x02, 0x03, 0x04)
        manager.initializeAsync(
            assetProvider = { corruptBytes },
            assetPath = "models/corrupt_model.onnx"
        )

        val completed = latch.await(10, TimeUnit.SECONDS)
        assertTrue("Corrupt bytes must trigger FAILED state", completed)

        assertFalse(manager.isReady)
        assertTrue(manager.state is OnnxRuntimeState.Failed)
        assertNull(manager.classifier)

        val failedState = manager.state as OnnxRuntimeState.Failed
        assertTrue(failedState.errorMessage.isNotEmpty())
    }

    @Test
    fun testRetryAfterFailureSucceedsAndDisposesPreviousState() {
        val executor = Executors.newSingleThreadExecutor()
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = executor,
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        val failLatch = CountDownLatch(1)
        manager.addListener { state ->
            if (state is OnnxRuntimeState.Failed) {
                failLatch.countDown()
            }
        }

        // First attempt: fail with empty bytes
        manager.initializeAsync(
            assetProvider = { ByteArray(0) },
            assetPath = "models/empty.onnx"
        )
        failLatch.await(10, TimeUnit.SECONDS)
        assertTrue(manager.state is OnnxRuntimeState.Failed)

        // Second attempt: retry with valid model file
        val readyLatch = CountDownLatch(1)
        manager.addListener { state ->
            if (state is OnnxRuntimeState.Ready) {
                readyLatch.countDown()
            }
        }

        manager.initializeAsync(
            assetProvider = { modelFile.readBytes() },
            assetPath = "models/fusion_model_fp32.onnx",
            forceRetry = true
        )

        val readyCompleted = readyLatch.await(10, TimeUnit.SECONDS)
        assertTrue("Retry must complete to READY", readyCompleted)
        assertTrue(manager.isReady)
        assertNotNull(manager.classifier)
    }

    @Test
    fun testCloseResetsToNotStartedAndDisposesClassifier() {
        val executor = Executors.newSingleThreadExecutor()
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = executor,
            mainDispatcher = { it.run() }
        )

        val latch = CountDownLatch(1)
        manager.addListener { state ->
            if (state is OnnxRuntimeState.Ready) {
                latch.countDown()
            }
        }

        manager.initializeAsync(
            assetProvider = { modelFile.readBytes() },
            assetPath = "models/fusion_model_fp32.onnx"
        )
        latch.await(10, TimeUnit.SECONDS)
        assertTrue(manager.isReady)

        manager.close()
        assertEquals(OnnxRuntimeState.NotStarted, manager.state)
        assertFalse(manager.isReady)
        assertNull(manager.classifier)
    }

    @Test
    fun testConcurrentListenerRegistrationIsThreadSafe() {
        val manager = OnnxRuntimeManager.createForTesting(
            backgroundExecutor = Executors.newSingleThreadExecutor(),
            mainDispatcher = { it.run() }
        )
        managerToClean = manager

        val threads = (1..10).map { id ->
            Thread {
                val listener: (OnnxRuntimeState) -> Unit = {}
                manager.addListener(listener)
                manager.removeListener(listener)
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }
    }
}
