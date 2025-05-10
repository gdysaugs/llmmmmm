// backend APIとの通信を行うサービス

interface ChatRequest {
  text: string;
}

interface ChatResponse {
  llm_response: string;
  audio_url?: string;
  error?: string;
}

interface TalkingFaceResponse {
  success: boolean;
  video_url?: string;
  details?: any;
  error?: string;
}

interface ChatWithTalkingFaceResponse {
  llm_response: string;
  audio_url?: string;
  video_url?: string;
  error?: string;
}

interface SadTalkerStatusResponse {
  available: boolean;
}

const API_BASE_URL = 'http://localhost:8000';

// チャットリクエストを送信
export const sendChatRequest = async (text: string): Promise<ChatResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'API request failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Chat request error:', error);
    return {
      llm_response: 'エラーが発生しました。もう一度やり直してください。',
      error: error instanceof Error ? error.message : '不明なエラー',
    };
  }
};

// 口パク動画生成リクエストを送信
export const generateTalkingFace = async (
  imageFile: File,
  audioPath: string,
  options: {
    poseStyle?: number;
    batchSize?: number;
    faceEnhancement?: boolean;
    stillMode?: boolean;
    useEnhancer?: boolean;
    preprocess?: string;
    enhancer?: string;
  } = {}
): Promise<TalkingFaceResponse> => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('audio_path', audioPath);
    
    // オプションパラメータを追加
    if (options.poseStyle !== undefined) formData.append('pose_style', options.poseStyle.toString());
    if (options.batchSize !== undefined) formData.append('batch_size', options.batchSize.toString());
    if (options.faceEnhancement !== undefined) formData.append('face_enhancement', options.faceEnhancement.toString());
    if (options.stillMode !== undefined) formData.append('still_mode', options.stillMode.toString());
    if (options.useEnhancer !== undefined) formData.append('use_enhancer', options.useEnhancer.toString());
    if (options.preprocess !== undefined) formData.append('preprocess', options.preprocess);
    if (options.enhancer !== undefined) formData.append('enhancer', options.enhancer);

    const response = await fetch(`${API_BASE_URL}/generate_talking_face`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'API request failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Generate talking face error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : '不明なエラー',
    };
  }
};

// チャットと口パク動画生成を一度に行うリクエストを送信
export const chatWithTalkingFace = async (
  text: string,
  imageFile: File,
  options: {
    poseStyle?: number;
    stillMode?: boolean;
  } = {}
): Promise<ChatWithTalkingFaceResponse> => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    // リクエストデータを追加
    const requestData = { text };
    formData.append('request', new Blob([JSON.stringify(requestData)], { type: 'application/json' }));
    
    // オプションパラメータを追加
    if (options.poseStyle !== undefined) formData.append('pose_style', options.poseStyle.toString());
    if (options.stillMode !== undefined) formData.append('still_mode', options.stillMode.toString());

    const response = await fetch(`${API_BASE_URL}/chat_with_talking_face`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'API request failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Chat with talking face error:', error);
    return {
      llm_response: 'エラーが発生しました。もう一度やり直してください。',
      error: error instanceof Error ? error.message : '不明なエラー',
    };
  }
};

// SadTalkerの状態を確認
export const checkSadTalkerStatus = async (): Promise<SadTalkerStatusResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/sadtalker_status`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch SadTalker status');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Check SadTalker status error:', error);
    return { available: false };
  }
}; 