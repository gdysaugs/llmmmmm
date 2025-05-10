import { useState, useRef } from 'react';

interface TalkingFaceUploaderProps {
  onImageSelected: (file: File) => void;
  disabled?: boolean;
}

const TalkingFaceUploader: React.FC<TalkingFaceUploaderProps> = ({ 
  onImageSelected, 
  disabled = false 
}) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 画像ファイルのみ許可
    if (!file.type.startsWith('image/')) {
      alert('画像ファイルを選択してください。');
      return;
    }

    // プレビュー用URLを作成
    const imageUrl = URL.createObjectURL(file);
    setPreviewUrl(imageUrl);
    
    // 親コンポーネントに選択された画像を通知
    onImageSelected(file);
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    
    const file = event.dataTransfer.files[0];
    if (!file) return;
    
    if (!file.type.startsWith('image/')) {
      alert('画像ファイルを選択してください。');
      return;
    }
    
    const imageUrl = URL.createObjectURL(file);
    setPreviewUrl(imageUrl);
    
    onImageSelected(file);
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <div
        className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50 ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
        onClick={disabled ? undefined : handleButtonClick}
        onDragOver={disabled ? undefined : handleDragOver}
        onDrop={disabled ? undefined : handleDrop}
      >
        {previewUrl ? (
          <div className="relative">
            <img
              src={previewUrl}
              alt="顔画像プレビュー"
              className="max-h-48 mx-auto rounded"
            />
            <button
              className={`mt-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 ${
                disabled ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              onClick={(e) => {
                e.stopPropagation();
                if (!disabled) handleButtonClick();
              }}
              disabled={disabled}
            >
              画像を変更
            </button>
          </div>
        ) : (
          <>
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M12 6v6m0 0v6m0-6h6m-6 0H6"
              ></path>
            </svg>
            <p className="mt-2 text-sm text-gray-600">
              クリックまたはドラッグ＆ドロップで顔画像をアップロード
            </p>
            <p className="text-xs text-gray-500 mt-1">
              PNG, JPG, GIF (最大 10MB)
            </p>
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
        disabled={disabled}
      />
    </div>
  );
};

export default TalkingFaceUploader; 