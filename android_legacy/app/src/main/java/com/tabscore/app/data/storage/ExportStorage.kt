package com.tabscore.app.data.storage

import android.content.Context
import android.os.Environment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

interface ExportStorage {
    suspend fun saveBytes(filename: String, bytes: ByteArray): String
}

class FileExportStorage(
    private val context: Context
) : ExportStorage {
    override suspend fun saveBytes(filename: String, bytes: ByteArray): String = withContext(Dispatchers.IO) {
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            ?: context.filesDir
        if (!dir.exists()) {
            dir.mkdirs()
        }
        val file = File(dir, filename)
        file.writeBytes(bytes)
        file.absolutePath
    }
}
