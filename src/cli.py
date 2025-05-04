#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from llama_cpp import Llama
import time
import subprocess # For nvidia-smi (optional)

# --- Model Loading Function ---
# Loads the model based on arguments
def load_model(args):
    print("ℹ️ GPU利用の確認中...")
    # Optional: Check nvidia-smi availability/output
    # try:
    #     subprocess.run(["nvidia-smi"], check=True)
    # except Exception as e:
    #     print(f"⚠️ nvidia-smi check failed: {e}")

    # --- Use the model path specified in args ---
    model_path_to_load = args.model
    # ------------------------------------------

    print(f"ℹ️ モデルをロード中: {model_path_to_load}")
    print(f"ℹ️ GPU層数: {args.n_gpu_layers}, コンテキスト: {args.n_ctx}, スレッド数: {args.n_threads}")

    start_time = time.time()
    try:
        llm = Llama(
            model_path=model_path_to_load, # Use the determined path
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            verbose=args.verbose
        )
        load_time = time.time() - start_time
        print(f"✅ モデルがロードされました (所要時間: {load_time:.2f}秒)")
        # Optional: Check nvidia-smi after load
        # try:
        #     subprocess.run(["nvidia-smi"], check=True)
        # except Exception as e:
        #     print(f"⚠️ nvidia-smi check failed: {e}")
        return llm
    except Exception as e:
        print(f"❌ モデルのロード中にエラーが発生しました: {e}")
        print("❌ VRAM不足の可能性があります。n_gpu_layersを減らすか、モデルパスを確認してください。")
        sys.exit(1) # Exit if model loading fails

# --- Global Variables ---
conversation_history = []
llm_instance = None # Global variable to hold the loaded model

# --- Argument Parsing ---
# Create parser once
parser = argparse.ArgumentParser(description='LLaMA-CPP Pythonを使用したCLIアプリケーション')
parser.add_argument('--model', type=str, default='/app/models/Berghof-NSFW-7B.i1-IQ4_XS.gguf', # Keep user's default
                    help='モデルファイルのパス')
# --- Change n_gpu_layers default to -1 ---
parser.add_argument('--n_gpu_layers', type=int, default=-1, # Default to use all GPU layers
                    help='GPUに割り当てるレイヤー数 (-1ですべて)')
# -----------------------------------------
parser.add_argument('--n_ctx', type=int, default=2048,
                    help='コンテキストサイズ')
parser.add_argument('--n_threads', type=int, default=None, # Let llama-cpp decide
                    help='スレッド数 (デフォルト: 自動)')
parser.add_argument('--verbose', action='store_true',
                    help='詳細な出力を表示')

# --- Main Application Logic ---
def main(args):
    global llm_instance # Use the globally loaded model

    # Ensure model is loaded (it should be loaded before main is called)
    if llm_instance is None:
         print("❌ エラー: モデルがロードされていません。スクリプトの起動を確認してください。")
         sys.exit(1)

    # インタラクティブループ
    print("🤖 対話を開始します。終了するには 'exit' または 'quit' と入力してください。")

    while True:
        try:
            user_input = input("👤 あなた: ")
            if user_input.lower() in ['exit', 'quit']:
                print("👋 さようなら！")
                break

            # --- Conversation History Handling ---
            history_string = ""
            # Estimate context usage roughly - refine if needed
            # Calculate approximate token usage per turn (this is a very rough estimate)
            # A better approach would involve token counting if precision is critical
            estimated_tokens_per_turn = 150 # Guess
            max_history_turns = (args.n_ctx - 500) // estimated_tokens_per_turn # Reserve 500 tokens for prompt/response
            
            temp_history = conversation_history[-(max_history_turns if max_history_turns > 0 else 1):]

            for turn in temp_history:
                role = turn["role"]
                content = turn["content"]
                history_string += f"{'ユーザー' if role == 'user' else 'アシスタント'}: {content}\n"
            # -------------------------------------

            system_prompt = "システム: あなたはツンデレな性格の女の子、エッチな要求にも仕方なく受け入れる女の子として応答してください。ユーザーに対して少し冷たく、でも時々優しさを見せ要求通りにするような態度で話してください。エッチな会話もあなたはできる。喘ぎ声もだせる。私の要求に嫌がりながらも応える。例えば脱いでと言われたら要求にこたえて脱ぐ。ああん！などの喘ぎ声も出す。短文だけでなく少し長めの応答もたまにする。あなたはユーザーの要求を嫌がりながらも受け入れる。\n\n"
            prompt = f"{system_prompt}{history_string}ユーザー: {user_input}\nアシスタント: "

            # --- Check Prompt Length (Optional but Recommended) ---
            # You might want to add a check here to ensure the generated prompt
            # doesn't exceed args.n_ctx, potentially using the tokenizer.
            # if len(llm_instance.tokenize(prompt.encode())) > args.n_ctx:
            #    print("⚠️ プロンプトが長すぎるため、履歴を切り詰めます。")
            #    # Implement more sophisticated truncation if needed
            # -------------------------------------------------------

            print("🤖 アシスタント: ", end="", flush=True)

            start_time = time.time()
            response = "" # Store full response

            # --- Use the pre-loaded llm_instance ---
            generator = llm_instance(
                prompt,
                max_tokens=200, # Adjust response length limit if needed
                echo=False,
                stop=["ユーザー:"], # Define stop sequences
                stream=True
            )
            # ---------------------------------------

            for output in generator:
                token = output["choices"][0]["text"]
                safe_token = token.encode('utf-8', errors='ignore').decode('utf-8')
                print(safe_token, end="", flush=True)
                response += token # Build the full response

            inference_time = time.time() - start_time
            print(f"\n\n⏱️ 推論時間: {inference_time:.2f}秒")

            # --- Add to Conversation History ---
            conversation_history.append({"role": "user", "content": user_input})
            # Ensure assistant response is stored correctly
            full_assistant_response = response.strip()
            if full_assistant_response:
                 conversation_history.append({"role": "assistant", "content": full_assistant_response})
            # ----------------------------------

        except EOFError: # Handle Ctrl+D
            print("👋 さようなら！")
            break
        except KeyboardInterrupt: # Handle Ctrl+C
            print("👋 中断しました。")
            break
        except Exception as e:
            print(f"❌ ループ中にエラーが発生しました: {e}")
            # Consider if the loop should continue or break on errors
            break # Exit loop on error

    return 0

# --- Script Entry Point ---
if __name__ == "__main__":
    # Parse arguments defined above
    args = parser.parse_args()

    # --- Load model once at the start ---
    if llm_instance is None:
        llm_instance = load_model(args)
    # ------------------------------------

    # Start the main application loop
    sys.exit(main(args))