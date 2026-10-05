"""Prueba del modo thinking de Gemma 4 en Ollama.

Dia 10. El hallazgo que justifica este script: en Gemma 4 el modo thinking
viene ACTIVADO por defecto, y con un limite bajo de tokens se come todo el
presupuesto ANTES de emitir contenido. El resultado es un message.content VACIO
con eval_count al tope.

Por eso el chat_dual.py manda think=false por defecto. Este script mide las dos
formas para que la decision no sea una opinion.

Uso:
    python test_thinking.py
    python test_thinking.py --tokens 300
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

HOST = "http://localhost:11434"
MODELO = "gemma4:e4b"
PROMPT = "Di hola en una frase."


def pedir(think: bool, num_predict: int, temperature: float = 0.7) -> dict:
    cuerpo = {
        "model": MODELO,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    peticion = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(cuerpo).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(peticion, timeout=1200) as r:
            datos = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8')[:200]}"}
    except urllib.error.URLError as e:
        return {"error": f"sin conexion: {e.reason}"}

    latencia = time.perf_counter() - t0
    gen = datos.get("eval_count") or 0
    eval_ns = datos.get("eval_duration") or 0
    texto = (datos.get("message") or {}).get("content", "")
    return {
        "think": think,
        "num_predict": num_predict,
        "eval_count": gen,
        "done_reason": datos.get("done_reason"),
        "tok_s": round(gen / (eval_ns / 1e9), 2) if eval_ns else None,
        "latencia_s": round(latencia, 2),
        "chars": len(texto),
        "vacio": len(texto.strip()) == 0,
        "texto": texto,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Mide think=true vs think=false en Gemma 4.")
    ap.add_argument("--tokens", type=int, default=256,
                    help="Limite de tokens a probar (por defecto 256).")
    args = ap.parse_args()

    print("=" * 92)
    print(f"MODE THINKING — {MODELO} · limite {args.tokens} tokens · prompt: {PROMPT!r}")
    print("=" * 92)

    muestras = []
    for think in (True, False):
        r = pedir(think, args.tokens)
        if "error" in r:
            print(f"\n  think={think}: ERROR {r['error']}")
            print("  Arranca Ollama con: ollama serve")
            return 2
        muestras.append(r)

        marca = "VACIO" if r["vacio"] else f"{r['chars']} chars"
        print(f"\n  think={str(think):<5} tokens={r['eval_count']:<5}"
              f" tok/s={str(r['tok_s']):<6} latencia={r['latencia_s']}s"
              f"  done={r['done_reason']}  -> {marca}")
        print(f"    {r['texto'][:200]}{'...' if len(r['texto']) > 200 else ''}")

    print()
    print("=" * 92)
    print("CONCLUSION")
    print("=" * 92)
    con, sin = muestras

    if con["vacio"] and not sin["vacio"]:
        print(f"  think=true  devuelve VACIO: {con['eval_count']} tokens consumidos y")
        print(f"  {con['chars']} caracteres utiles. El presupuesto se fue al")
        print("  razonamiento interno antes de emitir la respuesta.")
        print(f"  think=false responde en {sin['eval_count']} tokens y {sin['chars']} caracteres.")
        print()
        print("  => Por defecto hay que enviar think=false. Es lo que hace chat_dual.py.")
        return 0

    if con["vacio"] and sin["vacio"]:
        print("  AMBAS formas devuelven vacio. El limite de tokens es demasiado bajo:")
        print(f"  prueba con --tokens 600 o mas. Este NO es un fallo de thinking.")
        return 1

    print(f"  Con limite {args.tokens} ambos modos responden:")
    print(f"    think=true : {con['eval_count']} tokens, {con['chars']} chars")
    print(f"    think=false: {sin['eval_count']} tokens, {sin['chars']} chars")
    print()
    print("  A 256 tokens el modo thinking puede agotar el presupuesto. El umbral")
    print("  exacto depende del prompt: uno complejo razonara mas antes de responder.")
    print("  Por eso think=false es el valor por defecto, no una excepcion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
