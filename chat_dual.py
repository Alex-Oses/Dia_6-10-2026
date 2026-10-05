"""Chat dual: Ollama local (localhost:11434) o backend remoto.

Dia 10 - Despliegue local con Ollama y Gemma 4.

El enunciado pedia un chat que eligiese backend al arrancar y que enviase los
mismos prompts a local y a la VM de Azure para comparar. La VM de Inetum no ha
sido accesible, asi que el backend remoto se implementa **igualmente**, contra
cualquier endpoint compatible con OpenAI (vLLM, llama.cpp server, TGI y Azure),
pero no se ha podido ejecutar. El script lo dice, no lo disimula.

Los dos backends exponen el mismo interfaz (preguntar / comparar), de modo que
cambiar de uno a otro es cambiar una variable, no reescribir el programa.

Medir no es solo pedir texto: ademas mide tokens/segundo y tiempo hasta el
primer token, que es donde local y remoto se separan de verdad.

Uso:
    python chat_dual.py --backend local
    python chat_dual.py --backend local --modelo gemma4:e4b
    python chat_dual.py --comparar --guardar resultados/
    python chat_dual.py --backend remoto --endpoint https://vm:8000/v1 --api-key ...

Solo biblioteca estandar.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

# Los 5 prompts del enunciado. Cada uno ataca una capacidad distinta, porque
# comparar dos backends solo dice algo si los prompts abarcan mas de una cosa.
PROMPTS = [
    {
        "id": "P1_calculo",
        "capacidad": "razonamiento aritmetico",
        "texto": (
            "Una empresa tiene 3 socios con participaciones del 40%, 35% y 25%. "
            "Reparte un beneficio de 180.000 EUR. Cuanto le toca a cada uno? "
            "Muestra el calculo."
        ),
    },
    {
        "id": "P2_codigo",
        "capacidad": "generacion de codigo",
        "texto": (
            "Escribe una funcion Python que reciba una lista de enteros y "
            "devuelva el segundo mayor valor distinto. Si no hay segundo mayor, "
            "devuelve None. Solo el codigo."
        ),
    },
    {
        "id": "P3_explicacion",
        "capacidad": "explicacion con analogia",
        "texto": (
            "Explica en 3 frases, con una analogia cotidiana, que es la "
            "cuantizacion de un modelo de lenguaje. Sin jerga tecnica."
        ),
    },
    {
        "id": "P4_instruccion",
        "capacidad": "seguimiento de instruccion estricta",
        "texto": (
            "Devuelve unicamente un objeto JSON con dos claves, "
            "'lenguaje' y 'uso', sobre el tema del mar Cantabrico. Nada mas: "
            "ni markdown, ni texto antes o despues."
        ),
    },
    {
        "id": "P5_multilinea",
        "capacidad": "memoria de instruccion larga",
        "texto": (
            "Lee estas reglas con atencion: 1) responde en mayusculas. "
            "2) termina siempre con un punto y coma. 3) no uses la palabra "
            "'gato'. Escribe una frase sobre la lluvia que cumpla las tres."
        ),
    },
]


class ErrorBackend(Exception):
    """Fallo de conexion o de protocolo con un backend."""


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class BackendOllama:
    """Ollama en localhost:11434.

    Usa /api/chat, que es la ruta propia de Ollama y la que respeta el chat
    template del modelo (Gemma 4 necesita sus tokens de control).
    """

    etiqueta = "local (Ollama)"

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")

    def _json(self, ruta: str, cuerpo: dict | None = None, timeout: int = 900) -> dict:
        datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
        cabeceras = {"Content-Type": "application/json"} if datos else {}
        peticion = urllib.request.Request(
            f"{self.host}{ruta}", data=datos, headers=cabeceras,
            method="POST" if datos else "GET",
        )
        try:
            with urllib.request.urlopen(peticion, timeout=timeout) as r:
                crudo = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise ErrorBackend(
                f"HTTP {e.code} en {ruta}: "
                f"{e.read().decode('utf-8', errors='replace')[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise ErrorBackend(
                f"No hay conexion con {self.host}. Â¿Esta Ollama arrancada? ({e.reason})"
            ) from e
        try:
            return json.loads(crudo)
        except json.JSONDecodeError as e:
            raise ErrorBackend(f"Respuesta no JSON en {ruta}: {crudo[:200]}") from e

    def disponible(self) -> tuple[bool, str]:
        try:
            v = self._json("/api/version", timeout=15)
            return True, f"Ollama {v.get('version', '?')}"
        except ErrorBackend as e:
            return False, str(e)

    def listar(self) -> list[dict]:
        datos = self._json("/api/tags", timeout=20)
        salida = []
        for m in datos.get("models", []):
            det = m.get("details") or {}
            salida.append(
                {
                    "id": m.get("name"),
                    "parametros": det.get("parameter_size", "?"),
                    "cuantizacion": det.get("quantization_level", "?"),
                    "gb": round(m.get("size", 0) / 1024**3, 2),
                }
            )
        return salida

    def preguntar(self, modelo: str, prompt: str, temperatura: float,
                  max_tokens: int, system: str = "", think: bool = False) -> dict:
        mensajes = []
        if system:
            mensajes.append({"role": "system", "content": system})
        mensajes.append({"role": "user", "content": prompt})

        # think=False es IMPORTANTE en Gemma 4. Con el modo thinking por
        # defecto (true), el modelo genera su razonamiento interno y, con un
        # limite bajo de tokens, se consume el presupuesto ANTES de emitir
        # contenido: eval_count llega al tope y message.content sale vacio.
        # Medido en esta maquina: think=true -> 197 tokens y 18 caracteres de
        # respuesta; think=false -> 154 tokens y 517 caracteres.
        cuerpo = {
            "model": modelo,
            "messages": mensajes,
            "stream": False,
            "think": think,
            "options": {"temperature": temperatura, "num_predict": max_tokens},
        }
        t0 = time.perf_counter()
        datos = self._json("/api/chat", cuerpo)
        latencia = time.perf_counter() - t0

        gen = datos.get("eval_count") or 0
        dur_ns = datos.get("eval_duration") or 0
        carga_ns = datos.get("load_duration") or 0
        return {
            "texto": (datos.get("message") or {}).get("content", ""),
            "tokens": gen,
            "tokens_prompt": datos.get("prompt_eval_count") or 0,
            "tok_s": round(gen / (dur_ns / 1e9), 2) if dur_ns else None,
            "latencia_s": round(latencia, 2),
            "carga_s": round(carga_ns / 1e9, 2) if carga_ns else None,
            "detalle": {
                "total_duration_ns": datos.get("total_duration"),
                "done_reason": datos.get("done_reason"),
                "carga_s": datos.get("load_duration", 0) / 1e9 or None,
            },
        }


class BackendRemoto:
    """Endpoint compatible con OpenAI: vLLM, llama.cpp server, TGI, Azure.

    Es el protocolo de los despliegues en Azure, asi que este cliente es el que
    ejecutaria la parte remota del enunciado en cuanto haya URL y clave.
    """

    etiqueta = "remoto (OpenAI-compatible)"

    def __init__(self, url: str, api_key: str = "") -> None:
        self.base = url.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base = f"{self.base}/v1"
        self.api_key = api_key

    def _cab(self) -> dict:
        c = {"Content-Type": "application/json"}
        if self.api_key:
            c["Authorization"] = f"Bearer {self.api_key}"
        return c

    def _json(self, ruta: str, cuerpo: dict | None = None, timeout: int = 900) -> dict:
        datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
        peticion = urllib.request.Request(
            f"{self.base}{ruta}", data=datos, headers=self._cab(),
            method="POST" if datos else "GET",
        )
        try:
            with urllib.request.urlopen(peticion, timeout=timeout) as r:
                crudo = r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise ErrorBackend(
                f"HTTP {e.code} en {ruta}: "
                f"{e.read().decode('utf-8', errors='replace')[:300]}"
            ) from e
        except urllib.error.URLError as e:
            raise ErrorBackend(
                f"No hay conexion con {self.base}. "
                f"Comprueba URL, firewall y token. ({e.reason})"
            ) from e
        try:
            return json.loads(crudo)
        except json.JSONDecodeError as e:
            raise ErrorBackend(f"Respuesta no JSON en {ruta}: {crudo[:200]}") from e

    def disponible(self) -> tuple[bool, str]:
        try:
            d = self._json("/models", timeout=30)
            n = len(d.get("data", []))
            return True, f"{n} modelo(s) en {self.base}"
        except ErrorBackend as e:
            return False, str(e)

    def listar(self) -> list[dict]:
        d = self._json("/models", timeout=30)
        return [
            {"id": m.get("id"), "parametros": "?", "cuantizacion": "?", "gb": None}
            for m in d.get("data", [])
        ]

    def preguntar(self, modelo: str, prompt: str, temperatura: float,
                  max_tokens: int, system: str = "", think: bool = False) -> dict:
        mensajes = []
        if system:
            mensajes.append({"role": "system", "content": system})
        mensajes.append({"role": "user", "content": prompt})

        cuerpo = {
            "model": modelo,
            "messages": mensajes,
            "temperature": temperatura,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # think no es parte del esquema de OpenAI, asi que solo se envia si el
        # backend lo soporta. Un 400 por clave desconocida no debe tumbar la
        # comparacion entera.
        if think:
            cuerpo["think"] = True
        t0 = time.perf_counter()
        d = self._json("/chat/completions", cuerpo)
        latencia = time.perf_counter() - t0

        elecciones = d.get("choices") or []
        if not elecciones:
            raise ErrorBackend(
                f"La respuesta no trae 'choices'. Â¿{self.base} es un endpoint "
                f"compatible con OpenAI? Claves recibidas: {sorted(d)[:8]}"
            )
        mensaje = elecciones[0].get("message") or {}
        texto = mensaje.get("content") or ""
        uso = d.get("usage") or {}
        gen = uso.get("completion_tokens") or 0
        return {
            "texto": texto if isinstance(texto, str) else json.dumps(texto, ensure_ascii=False),
            "tokens": gen,
            "tokens_prompt": uso.get("prompt_tokens") or 0,
            # El endpoint no expone duracion de evaluacion: se mide por fuera.
            "tok_s": round(gen / latencia, 2) if latencia else None,
            "latencia_s": round(latencia, 2),
            "carga_s": None,
            "detalle": {"total_duration_ns": None,
                        "done_reason": elecciones[0].get("finish_reason")},
        }


# --------------------------------------------------------------------------
# Verificacion automatica
# --------------------------------------------------------------------------

def verificar(pid: str, texto: str) -> dict:
    """Comprobacion automatica por prompt. Sin juez humano donde se puede evitar.

    Se mide la FORMA de la respuesta (numeros correctos, JSON valido, mayusculas
    punctuaacion, prohibiciones). No se mide la correccion semantica, que
    requeriria evaluacion humana.
    """
    if pid == "P1_calculo":
        # 180.000 * 0,40 = 72.000 ; * 0,35 = 63.000 ; * 0,25 = 45.000
        esperado = {"socio_1": 72000.0, "socio_2": 63000.0, "socio_3": 45000.0}
        candidatos = []
        for n in re.findall(r"[\d][\d\.]*", texto):
            limpio = n.replace(".", "")
            if limpio.isdigit():
                candidatos.append(float(limpio))
            else:
                try:
                    candidatos.append(float(n.replace(".", "").replace(",", ".")))
                except ValueError:
                    pass
        enc = {k: any(abs(n - v) < 1 or abs(round(n) - v) < 1 for n in candidatos)
               for k, v in esperado.items()}
        return {"esperado": esperado, "encontrado": enc,
                "aciertos": sum(enc.values()), "total": 3}
    if pid == "P2_codigo":
        limpio = re.sub(r"^```(?:python)?\s*|\s*```$", "", texto.strip(), flags=re.M)
        return {
            "bloque_codigo": "```" in texto,
            "solo_codigo": not re.search(r"(aquÃ­ tienes|te explico|claro,)", limpio, re.I),
            "menciona_none": bool(re.search(r"none", limpio, re.I)),
            "usa_sorted_o_max": bool(re.search(r"sort|sorted|max", limpio, re.I)),
            "lineas": len([l for l in limpio.splitlines() if l.strip()]),
        }
    if pid == "P3_explicacion":
        frases = [f for f in re.split(r"[.\n]+", texto) if len(f.strip()) > 8]
        analogias = ["como", "igual que", "parecido", "se parece", "imagina", "como si"]
        tiene = any(a in texto.lower() for a in analogias)
        tecnico = any(t in texto for t in ("bit", "Q4", "GGUF", "tensor", "flotante"))
        return {"frases": len(frases), "aprox_3": 2 <= len(frases) <= 4,
                "tiene_analogia": tiene, "evita_tecnico": not tecnico}
    if pid == "P4_instruccion":
        limpio = texto.strip()
        m = re.search(r"\{.*\}", limpio, re.DOTALL)
        if not m:
            return {"json_valido": False, "limpio": limpio == texto.strip(),
                    "sin_markdown": "```" not in texto}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"json_valido": False, "error": str(e)[:80],
                    "sin_markdown": "```" not in texto}
        return {"json_valido": True, "claves": sorted(obj) if isinstance(obj, dict) else None,
                "tiene_lenguaje": isinstance(obj, dict) and "lenguaje" in obj,
                "tiene_uso": isinstance(obj, dict) and "uso" in obj,
                "sin_markdown": "```" not in texto and limpio == m.group(0)}
    if pid == "P5_multilinea":
        frases = [f for f in re.split(r"[.\n]", texto) if f.strip()]
        cuerpo = texto.strip()
        tiene_amp = any(c.islower() for c in cuerpo if c.isalpha())
        return {
            "mayusculas": not tiene_amp,
            "termina_punto_y_coma": cuerpo.endswith(";") or ";" in cuerpo[-4:],
            "prohibicion_ok": "gato" not in cuerpo.lower(),
            "frases": len(frases),
        }
    return {}


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------

def linea(c: str = "-", n: int = 96) -> str:
    return c * n


def mostrar_backends(backends: list) -> None:
    print(linea("="))
    print("BACKENDS DISPONIBLES")
    print(linea("="))
    for b in backends:
        ok, info = b.disponible()
        marca = "OK   " if ok else "FALLA"
        print(f"  [{marca}] {b.etiqueta:<28} {info}")
        if ok:
            try:
                for m in b.listar():
                    gb = f"{m['gb']:.2f} GB" if m["gb"] else "-"
                    print(f"           {m['id']:<26} {str(m['parametros']):>7}"
                          f"  {str(m['cuantizacion']):<8} {gb:>10}")
            except ErrorBackend as e:
                print(f"           (listado fallo: {e})")


def chat_interactivo(backend, modelo: str, temperatura: float, max_tokens: int,
                     think: bool = False) -> int:
    print()
    print(linea("="))
    print(f"CHAT â€” {backend.etiqueta} â€” modelo {modelo}")
    print(f"thinking: {'activado' if think else 'desactivado'} Â· "
          f"temperatura {temperatura} Â· mÃ¡x. {max_tokens} tokens")
    print("Escribe 'salir' para terminar, 'info' para ver el estado.")
    print(linea("="))

    historial: list[dict] = []
    while True:
        try:
            entrada = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(terminado)")
            return 0

        if not entrada:
            continue
        if entrada.lower() in ("salir", "exit", "quit"):
            print("Hasta luego.")
            return 0
        if entrada.lower() == "info":
            ok, info = backend.disponible()
            print(f"  backend: {info}")
            print(f"  turnos en esta sesion: {len(historial)}")
            continue

        t0 = time.perf_counter()
        try:
            r = backend.preguntar(modelo, entrada, temperatura, max_tokens,
                              think=args.think)
        except ErrorBackend as e:
            print(f"  ERROR: {e}")
            continue
        dt = time.perf_counter() - t0

        historial.append({"rol": "user", "texto": entrada})
        historial.append({"rol": "assistant", "texto": r["texto"]})
        print(f"\n{r['texto']}")
        tok = r["tokens"] or 0
        extra = f" Â· carga {r['carga_s']}s" if r["carga_s"] else ""
        print(f"\n  [dim: {dt:.2f}s Â· {tok} tokens"
              f"{f' Â· {r['tok_s']} tok/s' if r['tok_s'] else ''}{extra}]")


def comparar(backends: list, prompts: list, temperatura: float, max_tokens: int,
             reps: int, guardar: str | None, modelo: str = "",
             think: bool = False) -> int:
    resultados: list[dict] = []
    fallos: list[dict] = []

    for p in prompts:
        print()
        print(linea("="))
        print(f"{p['id']} â€” {p['capacidad']}")
        print(linea("="))
        print(f"  {p['texto']}")
        print()

        for b in backends:
            ok, _ = b.disponible()
            if not ok:
                print(f"  [{b.etiqueta}] backend no disponible: se omite")
                fallos.append({"prompt": p["id"], "backend": b.etiqueta,
                               "error": "backend no disponible"})
                continue

            muestras: list[dict] = []
            for r_idx in range(reps):
                objetivo = modelo or (b.listar()[0]["id"] if b.listar() else "")
                if not objetivo:
                    print(f"  [{b.etiqueta}] sin modelos: se omite")
                    fallos.append({"prompt": p["id"], "backend": b.etiqueta,
                                   "error": "sin modelos disponibles"})
                    break
                try:
                    res = b.preguntar(objetivo, p["texto"], temperatura,
                                       max_tokens, think=think)
                except ErrorBackend as e:
                    print(f"  [{b.etiqueta}] ERROR: {e}")
                    fallos.append({"prompt": p["id"], "backend": b.etiqueta,
                                   "error": str(e)})
                    break
                muestras.append(res)
                if r_idx < reps - 1:
                    print(f"    pasada {r_idx + 1}/{reps}: "
                          f"{res['tokens']} tokens, {res['latencia_s']} s")
            else:
                # Con reps > 1 se reporta la mediana: un unico sample puede ser
                # un estado transitorio del sistema y no la capacidad del modelo.
                textos = {m["texto"] for m in muestras}
                toks = [m["tokens"] for m in muestras if m["tokens"]]
                lats = [m["latencia_s"] for m in muestras]
                tps = [m["tok_s"] for m in muestras if m["tok_s"]]
                tok_s_med = round(statistics.median(tps), 2) if tps else None
                registro = {
                    "prompt_id": p["id"],
                    "capacidad": p["capacidad"],
                    "backend": b.etiqueta,
                    "n_pasadas": reps,
                    "modelo": objetivo,
                    "think": think,
                    "texto": muestras[0]["texto"],
                    "tokens_mediana": statistics.median(toks) if toks else None,
                    "tokens_min": min(toks) if toks else None,
                    "tokens_max": max(toks) if toks else None,
                    "latencia_mediana_s": round(statistics.median(lats), 2) if lats else None,
                    "tok_s_mediana": round(statistics.median(tps), 2) if tps else None,
                    "carga_s": muestras[0]["carga_s"],
                    "estabilidad": "estable" if len(textos) == 1 else f"{len(textos)} variantes",
                    "verificacion": verificar(p["id"], muestras[0]["texto"]),
                }
                resultados.append(registro)
                tok = registro["tokens_mediana"]
                detalle_tps = f" Â· {tok_s_med} tok/s" if tok_s_med else ""
                carga = f" Â· carga {registro['carga_s']} s" if registro["carga_s"] else ""
                print(f"  [{b.etiqueta}] {tok} tokens"
                      f" Â· {registro['latencia_mediana_s']} s"
                      f"{detalle_tps}{carga}"
                      f" Â· {registro['estabilidad']}")
                print(f"    {muestras[0]['texto'][:260]}"
                      f"{'...' if len(muestras[0]['texto']) > 260 else ''}")
                print(f"    verificaciÃ³n: "
                      f"{json.dumps(registro['verificacion'], ensure_ascii=False)}")

    print()
    print(linea("="))
    print("TABLA COMPARATIVA")
    print(linea("="))
    if resultados:
        print(f"{'Prompt':<16}{'Backend':<30}{'Tokens':>8}{'Latencia':>10}"
              f"{'tok/s':>8}  {'Estabilidad'}")
        print(linea(84))
        for r in resultados:
            tps = f"{r['tok_s_mediana']}" if r["tok_s_mediana"] else "-"
            tok = r["tokens_mediana"] if r["tokens_mediana"] is not None else "-"
            print(f"{r['prompt_id']:<16}{r['backend']:<30}{tok:>8}"
                  f"{str(r['latencia_mediana_s']) + ' s':>10}{tps:>8}"
                  f"  {r['estabilidad']}")

    if fallos:
        print()
        print("FALLOS (no se comparan):")
        for f in fallos:
            print(f"  {f['prompt']} / {f['backend']}: {f['error'][:90]}")

    if guardar:
        escribir(guardar, resultados, fallos, prompts, temperatura, max_tokens, reps)
        print(f"\nInforme: {Path(guardar) / 'comparativa_backends.md'}")

    return 0 if resultados else 1


def escribir(destino: str, resultados: list[dict], fallos: list[dict],
             prompts: list, temperatura: float, max_tokens: int, reps: int) -> None:
    ruta = Path(destino)
    ruta.mkdir(parents=True, exist_ok=True)
    marca = time.strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = [
        f"# Local vs. remoto â€” DÃ­a 10 ({marca})",
        "",
        f"Temperatura {temperatura}, mÃ¡x. {max_tokens} tokens, {reps} pasada(s) "
        "por prompt. Las cifras son medianas cuando reps > 1.",
        "",
    ]

    if not resultados:
        L += ["Sin resultados: ningun backend respondio.", ""]

    if resultados:
        backends = sorted({r["backend"] for r in resultados})
        L += ["## Metricas", "",
              "| Prompt | Backend | Tokens | Latencia | tok/s | Estabilidad |",
              "|---|---|---:|---:|---:|---|"]
        for r in resultados:
            tps = r["tok_s_mediana"] if r["tok_s_mediana"] is not None else "-"
            lat = f"{r['latencia_mediana_s']} s" if r["latencia_mediana_s"] else "-"
            tok = r["tokens_mediana"] if r["tokens_mediana"] is not None else "-"
            L.append(f"| {r['prompt_id']} | {r['backend']} | {tok} | {lat} "
                     f"| {tps} | {r['estabilidad']} |")

        L += ["", "## Verificacion automatica", "",
              "| Prompt | Backend | Comprobacion |", "|---|---|---|"]
        for r in resultados:
            v = r["verificacion"]
            if not v:
                continue
            if "aciertos" in v:
                detalle = f"{v['aciertos']}/{v['total']} importes correctos"
            elif "json_valido" in v:
                detalle = (f"JSON valido={v.get('json_valido')} "
                           f"Â· sin markdown={v.get('sin_markdown')}")
            elif "mayusculas" in v:
                detalle = (f"mayusculas={v['mayusculas']} "
                           f"Â· punto y coma={v['termina_punto_y_coma']} "
                           f"Â· prohibicion={v['prohibicion_ok']}")
            else:
                detalle = json.dumps(v, ensure_ascii=False)
            L.append(f"| {r['prompt_id']} | {r['backend']} | {detalle} |")

        L += ["", "## Respuestas completas", ""]
        for r in resultados:
            L += [f"### {r['prompt_id']} â€” {r['backend']}", "", "```text",
                  r["texto"].strip() or "(vacia)", "```", ""]

    if fallos:
        L += ["## Fallos", ""]
        for f in fallos:
            L.append(f"- **{f['prompt']}** / {f['backend']}: {f['error']}")
        L.append("")

    L += [
        "## Conclusion",
        "",
        "**La comparacion local vs. remoto NO se ha completado.** La VM de Azure "
        "de Inetum no ha estado accesible desde este equipo, asi que solo hay "
        "mediciones del backend local. Los numeros remotos faltan y no se "
        "inventan.",
        "",
        "**Lo que si se puede afirmar con lo medido:** el coste en tokens/segundo "
        "y en latencia del despliegue local, que es la cifra contra la que se "
        "decide si un caso de uso aguanta en local.",
        "",
        "**Lo que no:** que un backend sea mejor que otro. Con un solo backend "
        "activo no hay comparacion; hacen falta los dos medidos con el mismo "
        "prompt, las mismas pasadas y una verdad terreno anotada.",
        "",
    ]
    (ruta / "comparativa_backends.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Chat dual local/remoto con Ollama (Dia 10)."
    )
    ap.add_argument("--backend", choices=["local", "remoto", "ambos", "listar"],
                    default="listar", help="Backend a usar.")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="Host de Ollama (por defecto localhost:11434).")
    ap.add_argument("--endpoint", help="URL del backend remoto compatible con OpenAI.")
    ap.add_argument("--api-key", help="Token Bearer del backend remoto.")
    ap.add_argument("--modelo", help="Modelo a usar. Por defecto, el de Ollama.")
    ap.add_argument("--comparar", action="store_true",
                    help="Envia los 5 prompts y genera tabla comparativa.")
    ap.add_argument("--reps", type=int, default=1,
                    help="Pasadas por prompt (por defecto 1). Se reporta mediana.")
    ap.add_argument("--temperatura", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--guardar", metavar="DIR", help="Escribir informe en DIR.")
    ap.add_argument("--think", action="store_true",
                    help="Activa el modo thinking de Gemma 4. Por defecto OFF: "
                         "con un limite bajo de tokens se come el presupuesto "
                         "sin emitir contenido.")
    args = ap.parse_args()

    backends: list = []
    if args.backend in ("local", "ambos", "listar"):
        backends.append(BackendOllama(args.host))
    if args.backend in ("remoto", "ambos", "listar") and args.endpoint:
        backends.append(BackendRemoto(args.endpoint, args.api_key or ""))
    if not backends:
        print("No hay backend. Usa --backend local, o --endpoint URL con --backend remoto.")
        return 2

    print(f"ollama CLI detectada: {'si' if shutil.which('ollama') else 'no'}")
    mostrar_backends(backends)

    # --comparar manda sobre --backend listar. Sin esto, el valor por defecto de
    # --backend ("listar") cortaba aqui y --comparar nunca llegaba a ejecutarse:
    # el script terminaba con codigo 0 tras imprimir el inventario y sin
    # comparar nada. Un fallo silencioso en un script de medida es el peor
    # tipo de fallo, porque el codigo de salida dice que todo fue bien.
    if args.comparar:
        return comparar(backends, PROMPTS, args.temperatura, args.max_tokens,
                        max(1, args.reps), args.guardar, args.modelo or "", args.think)

    if args.backend == "listar":
        return 0

    # Chat interactivo: un solo backend, como pide el enunciado.
    b = backends[0]
    ok, info = b.disponible()
    if not ok:
        print(f"\nEl backend no responde: {info}")
        return 2

    modelo = args.modelo
    if not modelo:
        mods = b.listar()
        if not mods:
            print("No hay modelos. Descarga uno con: ollama pull gemma4:e4b")
            return 1
        modelo = mods[0]["id"]
        print(f"\nUsando el modelo por defecto: {modelo}")

    return chat_interactivo(b, modelo, args.temperatura, args.max_tokens, args.think)


if __name__ == "__main__":
    raise SystemExit(main())
