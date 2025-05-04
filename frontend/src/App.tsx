import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [inputText, setInputText] = useState<string>('')
  const [llmResponse, setLlmResponse] = useState<string>('')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!inputText.trim()) return

    setIsLoading(true)
    setError(null)
    setLlmResponse('')
    setAudioUrl(null)

    try {
      // --- Backend API endpoint --- 
      // Make sure this matches the backend service address from frontend's perspective
      // Inside docker-compose network, service name `backend` might be resolvable
      // But for browser access, it's localhost:8000 (as mapped in docker-compose)
      const apiUrl = 'http://localhost:8000/chat'
      
      const response = await axios.post<{ 
        llm_response: string;
        audio_url?: string;
        error?: string; 
      }>(
        apiUrl,
        { text: inputText },
        { headers: { 'Content-Type': 'application/json' } }
      )

      if (response.data.error) {
        setError(response.data.error)
        setLlmResponse('')
        setAudioUrl(null)
      } else {
        setLlmResponse(response.data.llm_response)
        // Construct full URL if needed, depends on how backend serves it
        // Assuming backend serves audio from root path /audio/
        setAudioUrl(response.data.audio_url ? `http://localhost:8000${response.data.audio_url}` : null)
        setError(null)
      }
    } catch (err: any) {
      console.error("API Error:", err)
      setError(err.response?.data?.detail || err.message || 'An unknown error occurred')
      setLlmResponse('')
      setAudioUrl(null)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-2xl bg-gray-800 p-8 rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold mb-6 text-center text-teal-400">ツンデレAIボイスチャット</h1>

        <form onSubmit={handleSubmit} className="mb-6">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="ここにメッセージを入力してね...別に、知りたくないけど。"
            className="w-full p-3 bg-gray-700 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 resize-none h-32 text-white placeholder-gray-400"
            disabled={isLoading}
          />
          <button
            type="submit"
            className={`w-full mt-4 px-4 py-2 rounded-md font-semibold transition-colors duration-200 ${isLoading ? 'bg-gray-600 cursor-not-allowed' : 'bg-teal-600 hover:bg-teal-700'}`}
            disabled={isLoading}
          >
            {isLoading ? '考え中...' : '送信 (べ、別にあんたのためじゃないんだからね！)'}
          </button>
        </form>

        {isLoading && (
          <div className="flex justify-center items-center my-4">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-teal-500"></div>
            <p className="ml-3">ちょっと待ちなさいよ...</p>
          </div>
        )}

        {error && (
          <div className="bg-red-800 border border-red-600 text-red-100 px-4 py-3 rounded relative mb-4" role="alert">
            <strong className="font-bold">エラーよ！</strong>
            <span className="block sm:inline"> {error}</span>
          </div>
        )}

        {llmResponse && (
          <div className="bg-gray-700 p-4 rounded-md mb-4 border border-gray-600">
            <h2 className="text-xl font-semibold mb-2 text-teal-300">AIの応答:</h2>
            <p className="whitespace-pre-wrap">{llmResponse}</p>
          </div>
        )}

        {audioUrl && (
          <div className="mt-4 text-center">
            <h3 className="mb-2 text-lg font-medium text-teal-300">生成された音声:</h3>
            <audio controls src={audioUrl} className="w-full">
              お使いのブラウザは音声再生に対応していません。
            </audio>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
