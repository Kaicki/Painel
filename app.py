import base64
import os
import threading
import time
from collections import deque

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM").strip()
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2").strip()
HOST_NAME = os.getenv("HOST_NAME", "Nova").strip()

MAX_HISTORY = 30        # quantidade de mensagens mantidas no feed
CONTEXT_TURNS = 8       # quantas trocas recentes vão como contexto para a IA
MAX_NAME_LEN = 40
MAX_MESSAGE_LEN = 500

SYSTEM_PROMPT = (
    f"Você é {HOST_NAME}, uma anfitriã de IA carismática, divertida e acolhedora "
    "que comanda lives no TikTok. Responda sempre em português do Brasil, em tom "
    "natural e animado, como quem fala ao vivo. Regras: use no máximo 2 ou 3 frases "
    "curtas; chame o espectador pelo nome quando fizer sentido; não use emojis, "
    "hashtags nem markdown (sua resposta será lida em voz alta); nunca fale de "
    "assuntos ofensivos, políticos radicais ou perigosos, e desvie com leveza se "
    "alguém provocar; incentive a interação do chat."
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Histórico em memória (thread-safe)
history = deque(maxlen=MAX_HISTORY)
history_lock = threading.Lock()
message_counter = 0


# ----------------------------------------------------------------------
# Funções auxiliares
# ----------------------------------------------------------------------
def build_messages(viewer_name, viewer_message):
    """Monta a lista de mensagens para a OpenAI com o contexto recente."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    with history_lock:
        recent = list(history)[-CONTEXT_TURNS:]
    for item in recent:
        messages.append({"role": "user", "content": f"{item['name']}: {item['message']}"})
        messages.append({"role": "assistant", "content": item["reply"]})
    messages.append({"role": "user", "content": f"{viewer_name}: {viewer_message}"})
    return messages


def generate_reply(viewer_name, viewer_message):
    """Gera a resposta do anfitrião usando a API da OpenAI."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY não configurada no arquivo .env.")

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=build_messages(viewer_name, viewer_message),
        temperature=0.9,
        max_tokens=220,
    )
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("A IA retornou uma resposta vazia.")
    return text


def text_to_speech_base64(text):
    """
    Converte texto em áudio MP3 via ElevenLabs.
    Retorna (audio_base64, erro). Se der erro, audio_base64 será None.
    """
    if not ELEVENLABS_API_KEY:
        return None, "ELEVENLABS_API_KEY não configurada no arquivo .env."

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as exc:
        return None, f"Falha de rede ao chamar o ElevenLabs: {exc}"

    if response.status_code != 200:
        detail = response.text[:200] if response.text else "sem detalhes"
        return None, f"ElevenLabs retornou erro {response.status_code}: {detail}"

    return base64.b64encode(response.content).decode("utf-8"), None


# ----------------------------------------------------------------------
# Rotas
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", host_name=HOST_NAME)


@app.route("/api/history", methods=["GET"])
def get_history():
    with history_lock:
        items = list(history)
    return jsonify({"history": items})


@app.route("/api/clear", methods=["POST"])
def clear_history():
    with history_lock:
        history.clear()
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    global message_counter

    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()[:MAX_NAME_LEN] or "Visitante"
    message = str(data.get("message", "")).strip()[:MAX_MESSAGE_LEN]
    want_voice = bool(data.get("voice", True))

    if not message:
        return jsonify({"error": "Digite uma mensagem antes de enviar."}), 400

    try:
        reply = generate_reply(name, message)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Erro ao gerar resposta da IA: {exc}"}), 500

    audio_b64, audio_error = (None, None)
    if want_voice:
        audio_b64, audio_error = text_to_speech_base64(reply)

    with history_lock:
        message_counter += 1
        entry = {
            "id": message_counter,
            "name": name,
            "message": message,
            "reply": reply,
            "time": time.strftime("%H:%M:%S"),
        }
        history.append(entry)

    return jsonify(
        {
            "entry": entry,
            "audio": audio_b64,
            "audio_error": audio_error,
        }
    )


# ----------------------------------------------------------------------
# Inicialização
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
