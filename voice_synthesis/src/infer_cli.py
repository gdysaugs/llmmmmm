import argparse
import os
import torch
import soundfile as sf
from style_bert_vits2.tts_model import TTSModel  # style_bert_vits2 がインストールされている前提
import traceback # エラー表示用にインポート

def main(args):
    # デバイス設定 (GPU優先)
    if torch.cuda.is_available() and not args.cpu:
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # モデルのロード
    print(f"Loading model from: {args.model_dir}")
    # 具体的なモデルファイル名を指定 (chupa_1 の場合)
    # TODO: このファイル名はモデルによって変わる可能性があるので、
    # 本来はディレクトリ内の .safetensors を探すか、引数で指定する方が良い
    model_filename = "Anneli-nsfw_e300_s5100.safetensors" # 新しいモデル名に変更
    model_path = os.path.join(args.model_dir, model_filename)
    config_path = os.path.join(args.model_dir, "config.json")
    style_vec_path = os.path.join(args.model_dir, "style_vectors.npy")

    # ファイル存在チェック
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_filename} not found in {args.model_dir}")
        return
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {args.model_dir}")
        return
    if not os.path.exists(style_vec_path):
         print(f"Warning: style_vectors.npy not found in {args.model_dir}. Check if it's required or optional.")

    try:
        # 修正点: model_path に .safetensors ファイルへのフルパスを渡す
        model = TTSModel(
            model_path=model_path, # 具体的な .safetensors ファイルのパス
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=device
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    # 出力ディレクトリ作成
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # 音声合成の実行
    print(f"Synthesizing text: \"{args.text}\"")
    try:
        # style_bert_vits2 ライブラリの推論メソッドを呼び出す
        # 必要な引数はライブラリの仕様に合わせる (例)
        # style_reference_path や speaker_id などが必要な場合がある
        # style_weight などのパラメータも必要に応じて追加
        sample_rate, audio_data = model.infer(
            text=args.text,
            language=args.language,
            speaker_id=args.speaker_id,      # speaker_id を渡す
            style_weight=args.style_weight,  # style_weight を渡す
            # style_reference_path=args.style_ref, # 必要であれば追加
            # style_weight=args.style_weight,       # 必要であれば追加
            # sd_ratio=args.sd_ratio,             # 必要であれば追加
            # noise=args.noise,                 # 必要であれば追加
            # noise_scale=args.noise_scale,     # 必要であれば追加
            # speed=args.speed,                 # 必要であれば追加
        )
        print(f"Synthesis successful.")

        # --- デバッグ出力追加 ---
        print(f"Type of audio_data: {type(audio_data)}")
        if hasattr(audio_data, 'shape'):
            print(f"Shape of audio_data: {audio_data.shape}")
        else:
            print(f"audio_data does not have a shape attribute. Value: {audio_data}")
        print(f"Type of sample_rate: {type(sample_rate)}")
        if hasattr(sample_rate, 'shape'):
             print(f"Shape of sample_rate: {sample_rate.shape}")
        else:
             print(f"sample_rate does not have a shape attribute. Value: {sample_rate}")
        # --- デバッグ出力追加ここまで ---

        print(f"Saving to {args.output_path}")
        sf.write(args.output_path, audio_data, sample_rate)
        print("Audio saved.")
    except Exception as e:
        print(f"Error during synthesis or saving: {e}")
        traceback.print_exc() # 合成エラー時もトレースバック表示

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style-Bert-VITS2 CLI Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing model files (.safetensors, config.json, style_vectors.npy).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated audio file (.wav).")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize.")
    parser.add_argument("--language", type=str, default="JP", help="Language of the text (e.g., JP, EN, ZH).")
    # --- ↓↓↓ 引数追加 ↓↓↓ ---
    parser.add_argument("--speaker_id", type=int, default=0, help="Speaker ID (if model supports multiple speakers).")
    parser.add_argument("--style_weight", type=float, default=0.7, help="Weight for the style reference (style_vectors.npy).")
    # --- ↑↑↑ 引数追加 ↑↑↑ ---
    # --- オプション引数 (ライブラリの仕様に応じて追加・修正) ---
    # parser.add_argument("--style_ref", type=str, default=None, help="Path to the reference audio or style vector file for style control.") # style_vectors.npy を使うので不要か？
    # parser.add_argument("--sd_ratio", type=float, default=0.5, help="SDP ratio for controlling stochasticity.")
    # parser.add_argument("--noise", type=float, default=0.6, help="Noise level.")
    # parser.add_argument("--noise_scale", type=float, default=0.9, help="Noise scale.")
    # parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument("--cpu", action="store_true", help="Force use CPU even if GPU is available.")

    args = parser.parse_args()
    main(args) 