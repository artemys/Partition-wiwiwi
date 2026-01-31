package com.tabscore.app.data.storage

import android.content.ContentResolver
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

interface AudioReader {
    suspend fun readBytes(uriString: String): AudioPayload
}

class ContentResolverAudioReader(
    private val context: Context
) : AudioReader {
    override suspend fun readBytes(uriString: String): AudioPayload = withContext(Dispatchers.IO) {
        val uri = Uri.parse(uriString)
        val resolver: ContentResolver = context.contentResolver
        val mimeType = resolver.getType(uri) ?: "audio/*"
        val name = resolver.query(uri, null, null, null, null)?.use { cursor ->
            val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (cursor.moveToFirst() && nameIndex >= 0) cursor.getString(nameIndex) else null
        } ?: "audio"
        val bytes = resolver.openInputStream(uri)?.use { it.readBytes() }
            ?: error("Impossible de lire le fichier audio")
        AudioPayload(bytes = bytes, filename = name, mimeType = mimeType)
    }
}

data class AudioPayload(
    val bytes: ByteArray,
    val filename: String,
    val mimeType: String
)
