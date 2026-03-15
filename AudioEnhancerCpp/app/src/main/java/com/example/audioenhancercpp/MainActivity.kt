package com.example.audioenhancercpp

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.example.audioenhancercpp.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    
    // File Picker Launcher
    private val pickFileLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                processSelectedFile(uri)
            }
        }
    }

    companion object {
        init {
            System.loadLibrary("audioenhancercpp")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 1. Prepare Model on Launch
        val modelPath = getModelPath()
        if (modelPath.isEmpty()) {
            binding.sampleText.text = "Error: Could not load ONNX model."
            binding.btnPickFile.isEnabled = false
        } else {
            binding.sampleText.text = "Model Loaded. Ready."
        }

        // 2. Setup Button
        binding.btnPickFile.setOnClickListener {
            openFilePicker()
        }
    }

    private fun openFilePicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "audio/*" // Broader filter to ensure files are clickable
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("audio/wav", "audio/x-wav", "audio/wave"))
        }
        pickFileLauncher.launch(intent)
    }

    private fun processSelectedFile(uri: Uri) {
        binding.sampleText.text = "Reading file..."
        binding.btnPickFile.isEnabled = false

        CoroutineScope(Dispatchers.IO).launch {
            try {
                // 1. Read WAV bytes
                val inputBytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                if (inputBytes == null) {
                    throw IOException("Could not read file")
                }

                // 2. Parse WAV to FloatArray (Simple PCM 16-bit Mono parser)
                // Note: robustness depends on file format. This is a basic implementation.
                val inputFloats = wavToFloats(inputBytes)
                
                withContext(Dispatchers.Main) {
                     binding.sampleText.text = "Processing ${inputFloats.size} samples..."
                }

                // 3. Process via JNI
                val modelPath = getModelPath()
                val outputFloats = processAudioJNI(modelPath, inputFloats)

                // 4. Save Output to Public Downloads
                // Convert back to bytes first
                val outputWavBytes = floatsToWav(outputFloats)
                
                val filename = "enhanced_output_${System.currentTimeMillis()}.wav"
                val savedUri = saveToDownloads(filename, outputWavBytes)

                withContext(Dispatchers.Main) {
                    if (savedUri != null) {
                        binding.sampleText.text = "Success!\nSaved to Downloads: $filename"
                        Toast.makeText(this@MainActivity, "Saved to Downloads", Toast.LENGTH_LONG).show()
                    } else {
                        binding.sampleText.text = "Error: Could not save file."
                    }
                    binding.btnPickFile.isEnabled = true
                }

            } catch (e: Exception) {
                Log.e("AudioProcess", "Error", e)
                withContext(Dispatchers.Main) {
                    binding.sampleText.text = "Error: ${e.message}"
                    binding.btnPickFile.isEnabled = true
                }
            }
        }
    }

    private fun saveToDownloads(filename: String, bytes: ByteArray): Uri? {
        val contentValues = android.content.ContentValues().apply {
            put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, filename)
            put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "audio/wav")
            put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, android.os.Environment.DIRECTORY_DOWNLOADS)
        }

        val resolver = applicationContext.contentResolver
        val uri = resolver.insert(android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, contentValues)

        return if (uri != null) {
            try {
                resolver.openOutputStream(uri)?.use { it.write(bytes) }
                uri
            } catch (e: IOException) {
                Log.e("SaveFile", "Error writing to MediaStore", e)
                null
            }
        } else {
            null
        }
    }

    /**
     * EXTRACTS PCM DATA FROM WAV (Skips headers 44 bytes assumption)
     * For production, use a real WAV parser library.
     */
    /**
     * EXTRACTS PCM DATA FROM WAV (Robust Stereo/Mono Parser)
     * Reads header to determine channel count.
     * Returns: FloatArray (Interleaved L, R, L, R...)
     */
    private fun wavToFloats(wavBytes: ByteArray): FloatArray {
        // Minimum WAV header is 44 bytes
        if (wavBytes.size < 44) return FloatArray(0)

        val buffer = ByteBuffer.wrap(wavBytes).order(ByteOrder.LITTLE_ENDIAN)
        
        // Offset 22: NumChannels (2 bytes)
        val numChannels = buffer.getShort(22).toInt()
        // Offset 24: SampleRate (4 bytes)
        val sampleRate = buffer.getInt(24)
        // Offset 34: BitsPerSample (2 bytes)
        val bitsPerSample = buffer.getShort(34).toInt()

        if (sampleRate != 16000) {
            Log.w("WavParser", "Warning: Sample rate is $sampleRate, expected 16000")
        }

        // Jump to data chunk
        var pos = 12
        while (pos < wavBytes.size - 8) {
            val chunkId = String(wavBytes, pos, 4)
            val chunkSize = buffer.getInt(pos + 4)
            if (chunkId == "data") {
                pos += 8
                break
            }
            pos += 8 + chunkSize
        }
        
        // Start reading data
        buffer.position(pos)
        val dataSize = buffer.remaining()
        
        // Assuming 16-bit PCM for simplicity as per user req
        val numSamples = dataSize / 2
        val shortBuffer = buffer.asShortBuffer()
        
        // Target: Stereo Interleaved Vector (L0, R0, L1, R1 ...)
        // If Input is Mono: L0 -> L0, R0=L0
        // If Input is Stereo: L0, R0 -> L0, R0
        
        val targetSize = if (numChannels == 1) numSamples * 2 else numSamples
        val output = FloatArray(targetSize)
        
        var outIdx = 0
        var inIdx = 0
        
        if (numChannels == 1) {
            // Mono -> Stereo (Duplicate)
            while (shortBuffer.hasRemaining()) {
                val sample = shortBuffer.get() / 32768.0f
                output[outIdx++] = sample // L
                output[outIdx++] = sample // R
            }
        } else if (numChannels == 2) {
            // Stereo -> Stereo (Copy)
            while (shortBuffer.hasRemaining()) {
                output[outIdx++] = shortBuffer.get() / 32768.0f
            }
        } else {
            // Multi-channel (>2) -> Take first 2? Or Error. 
            // Lets take first 2.
            while (shortBuffer.hasRemaining()) {
                if (shortBuffer.remaining() >= numChannels) {
                    output[outIdx++] = shortBuffer.get() / 32768.0f // Ch1
                    output[outIdx++] = shortBuffer.get() / 32768.0f // Ch2
                    // Skip rest
                    for (k in 2 until numChannels) shortBuffer.get()
                } else {
                    break
                }
            }
        }
        
        return output
    }

    /**
     * CONVERTS FLOAT ARRAY TO WAV BYTES (16-bit PCM)
     */
    private fun floatsToWav(floats: FloatArray): ByteArray {
        val byteBuffer = ByteBuffer.allocate(44 + floats.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        
        // 1. Write Header
        val sampleRate = 16000
        val channels = 1
        val byteRate = sampleRate * channels * 2
        val totalDataLen = floats.size * 2
        val totalFileLen = totalDataLen + 36

        byteBuffer.put("RIFF".toByteArray())
        byteBuffer.putInt(totalFileLen)
        byteBuffer.put("WAVE".toByteArray())
        byteBuffer.put("fmt ".toByteArray())
        byteBuffer.putInt(16) // Subchunk1Size
        byteBuffer.putShort(1) // AudioFormat 1=PCM
        byteBuffer.putShort(channels.toShort())
        byteBuffer.putInt(sampleRate)
        byteBuffer.putInt(byteRate)
        byteBuffer.putShort(2) // BlockAlign
        byteBuffer.putShort(16) // BitsPerSample
        byteBuffer.put("data".toByteArray())
        byteBuffer.putInt(totalDataLen)

        // 2. Write Data
        for (f in floats) {
            // Clip
            var sample = f
            if (sample > 1.0f) sample = 1.0f
            if (sample < -1.0f) sample = -1.0f
            val shortSample = (sample * 32767).toInt().toShort()
            byteBuffer.putShort(shortSample)
        }

        return byteBuffer.array()
    }

    private fun getModelPath(): String {
        val modelName = "model.onnx"
        val file = File(filesDir, modelName)
        if (!file.exists()) {
            try {
                assets.open(modelName).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            } catch (e: IOException) {
                e.printStackTrace()
                return ""
            }
        }
        return file.absolutePath
    }

    external fun stringFromJNI(): String
    external fun processAudioJNI(modelPath: String, inputData: FloatArray): FloatArray
}