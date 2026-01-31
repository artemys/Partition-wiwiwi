package com.tabscore.app.data.remote

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request

class OkHttpFileDownloader(
    private val client: OkHttpClient
) : FileDownloader {
    override suspend fun download(url: String): ByteArray = withContext(Dispatchers.IO) {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("Téléchargement échoué: ${response.code}")
            }
            response.body?.bytes() ?: error("Fichier vide")
        }
    }
}
