# Día 10 — Despliegue local: Ollama y Gemma 4

**Martes 15 de septiembre de 2026** · Práctica Inetum

Entregable: **script de chat dual (local + VM de Azure)** y **informe
comparativo de rendimiento y calidad**. La parte remota está **implementada pero
no ejecutada**: la VM de Azure de Inetum no ha sido accesible desde este equipo.
No se inventan cifras remotas. Lo que sí se midió de verdad está en
`resultados/`, y el §7 dice exactamente qué se puede y qué no se puede concluir.

---

## Índice

1. [Instalación y despliegue de Gemma 4](#1-instalación-y-despliegue-de-gemma-4)
2. [Verificación por CLI y por API REST](#2-verificación-por-cli-y-por-api-rest)
3. [El hallazgo del día: el modo thinking](#3-el-hallazgo-del-dia-el-modo-thinking)
4. [Script de chat dual](#4-script-de-chat-dual)
5. [Resultados medidos](#5-resultados-medidos)
6. [Local vs. nube: el razonamiento](#6-local-vs-nube-el-razonamiento)
7. [Lo que no se puede concluir](#7-lo-que-no-se-puede-concluir)
8. [Errores encontrados](#8-errores-encontrados)
9. [Reproducir](#9-reproducir)
10. [Fuentes](#10-fuentes)

---

## 1. Instalación y despliegue de Gemma 4

### El equipo

```
Windows 11 · Intel Core i5-1135G7 @ 2,40 GHz · 4 núcleos (8 hilos)
RAM 15,7 GB (de los cuales ~2 GB reservados para la GPU integrada)
Disco 374,8 GB libres
GPU: Intel Iris Xe INTEGRADA, sin VRAM dedicada
Ollama 0.35.0
```

**No hay GPU dedicada.** Esto no es un detalle, es la condición que determina
toda la sección 5: en CPU la velocidad la marca el **ancho de banda de la
memoria**, no el número de núcleos.

### Instalación

Ollama ya estaba instalado (0.35.0). En Windows se instala desde
<https://ollama.com/download> y arranca como servicio. Verificado:

```powershell
ollama --version          # ollama version is 0.35.0
curl http://localhost:11434/api/version
# {"version":"0.35.0"}
```

### Descargar Gemma 4

```powershell
ollama pull gemma4:e4b
```

Descarga real: **6,6 GB**, verificada contra la ficha oficial de Ollama. El tag
`gemma4:e4b` corresponde a la cuantización **Q4_K_M**.

**Detalle de la descarga que conviene conocer:** tardó ~13 minutos a una velocidad
inicial de 716 KB/s, y aceleró hasta 32 MB/s al final (el arranque de la red en
este equipo va lento, no la descarga en sí). Los logs de `ollama pull` van a
`stderr`, no a `stdout`, y usan secuencias de control ANSI que ensucian cualquier
redirección a fichero.

### Qué se descargó exactamente

`ollama show gemma4:e4b`:

```
Model
    architecture        gemma4
    parameters          7.5B
    context length      131072
    embedding length    2560
    quantization        Q4_K_M

Capabilities
    completion   vision   audio   tools   thinking
        levels   false, true
        default  true              <-- el modo thinking viene ACTIVADO

Projector
    architecture        clip
    parameters          478.09M
    dimensions          2560
```

Cuatro cosas que solo se ven mirando el modelo desplegado:

| Campo | Valor | Por qué importa |
|---|---|---|
| `quantization` | Q4_K_M | **No es `Q4_0`.** La ficha de Ollama lista `gemma4:e4b` como Q4_0, pero el tag `e4b` (sin sufijo) se sirve como Q4_K_M, que es el preset recomendado. La diferencia: Q4_K_M reparte bits de forma desigual según la sensibilidad de cada tensor. |
| `parameters` | **7,5 B** | Ni los 4,5 B efectivos ni los 8,00 B totales de la ficha de Google. Es el total real del checkpoint cuantizado. |
| `thinking default` | **true** | El origen del §3. |
| `context length` | 131072 | 128K declarados, como en la ficha. Ojo: es el máximo teórico, no el que se puede usar aquí (ver §6). |

### El projector multimodal

`Projector: 478,09M parámetros, arquitectura clip`. Son **478 millones de
parámetros aparte del LLM**, y es lo que traduce imagen y audio al espacio de
embeddings del modelo.

Consecuencia práctica: los 6,6 GB de `gemma4:e4b` **no son solo el LLM**. Son
LLM cuantizado + proyector + KV cache. Por eso el modelo entra en 8 GB de RAM
justo, y por eso en el `ollama list` aparecen 6,13 GB (el `.gguf` del LLM) más lo
que el proyector ocupa aparte.

> Esto corrige una lectura simplista de la tabla del Día 9: E4B «necesita 4,5 GB»
> en Q4_0. Cierto para los pesos, pero para usar imagen o audio hay que añadir el
> proyector. La cifra honesta de despliegue es la que da `ollama list`.

### Modelos disponibles tras el despliegue

```
NAME           ID              SIZE      MODIFIED
gemma4:e4b     dc35e8d9c606    6.6 GB    (recién descargado)
llama3.1:8b    46e0c10c039e    4.9 GB    (del Día 9)
```

Los dos son Q4_K_M. `llama3.1:8b` se conserva como contraste del §5.

### Si el portátil no llegara: alternativas

El enunciado avisa de esta posibilidad. La tabla de tags de Ollama da las
opciones verificadas:

| Tag | Descarga | Cuándo |
|---|---:|---|
| `gemma4:e4b` | 6,6 GB | Con 8 GB de RAM, justo |
| `gemma4:e2b` | 4,6 GB | Con 6 GB. 2,3 B efectivos: pierde audio y vision |
| `gemma4:12b` | 7,7 GB | GPU dedicada de 12 GB |
| `gemma4:26b-a4b` | 16,0 GB | Servidor. MoE: 3,8 B activos |
| `gemma4:31b` | 19,0 GB | GPU de 24 GB |
| `gemma4:e4b-it-q8` | ~8 GB | Si hay RAM de sobra: calidad máxima |
| `gemma4:e4b-mlx` | — | **Solo Apple Silicon.** No aplica en Windows |
| `gemma4:e4b-nvfp4` | — | **Solo NVIDIA Blackwell.** Este equipo no es NVIDIA |

Los sufijos también enseñan algo: `-it` = instruction-tuned, `-bf16` = sin
cuantizar, `-mlx` = formato de Apple, `-nvfp4`/`-mxfp8` = formatos NVIDIA,
`-qat` = cuantizado con quantization-aware training.

---

## 2. Verificación por CLI y por API REST

El enunciado pide verificar que el modelo responde **por la CLI y por la API
REST**. Las dos vías, comprobadas.

### Vía CLI

```powershell
"Di hola en una frase." | ollama run gemma4:e4b
```

### Vía API REST

`POST http://localhost:11434/api/chat`:

```powershell
$body = @{
  model = "gemma4:e4b"
  messages = @(@{ role = "user"; content = "Di hola en una frase." })
  stream = $false
  think  = $false
  options = @{ temperature = 0; num_predict = 64 }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod http://localhost:11434/api/chat -Method POST -Body $body
```

La respuesta trae lo que hace falta para medir, en la misma llamada:

| Campo | Significado |
|---|---|
| `message.content` | El texto de la respuesta |
| `eval_count` | Tokens generados |
| `eval_duration` | Nanosegundos de generación → de aquí sale el **tok/s** |
| `prompt_eval_count` | Tokens del prompt (cuenta el coste de contexto) |
| `load_duration` | Tiempo de carga del modelo. **0,03 s** una vez resident |
| `total_duration` | Tiempo total |
| `done_reason` | `stop` (terminó) o `length` (se agotaron los tokens) |

**El `done_reason` es el campo que más evitó un error falso**, y está en el §3.

### Diferencia real entre CLI y API

La CLI usa `POST /api/generate` con un prompt plano; la API usa `POST /api/chat`
con mensajes roles. Se eligió `/api/chat` en el script del entregable porque
**respeta el chat template del modelo**, y Gemma 4 lo necesita: sus tokens de
control (`<|think|>`) y su rol de sistema nativo no funcionan igual con un prompt
suelto.

---

## 3. El hallazgo del día: el modo thinking

Este es el resultado más útil de la sesión, y nació de un fallo que parecía de
script.

### El síntoma

Primera petición de prueba con `num_predict=32`:

```
tokens generados : 32
latencia         : 49,45 s
done_reason      : length
--- CONTENIDO ---
(vacío)
```

**32 tokens consumidos, medio minuto de cálculo, y ni una palabra de respuesta.**
Con `num_predict=64` igual. Con 400 tokens, igual.

### La causa

`ollama show gemma4:e4b` dice:

```
Capabilities
    thinking
        levels   false, true
        default  true
```

En Gemma 4 el **modo thinking viene activado por defecto**. El modelo genera su
razonamiento interno antes de responder, y ese razonamiento consume el presupuesto
de tokens igual que la respuesta. Con un límite bajo, se lo come entero y no llega
a emitir nada: `eval_count` llega al tope, `done_reason` es `length` y
`message.content` queda vacío.

Un límite de 400 tokens no es exótico: es un valor razonable que alguien elegiría
para una prueba rápida.

### La solución

El campo `think` en el cuerpo de la petición. **Por defecto `false` en
`chat_dual.py`.**

### Lo medido

Mismo prompt, mismo modelo, límite de 128 tokens:

| `think` | Tokens | tok/s | Latencia | Contenido |
|---|---:|---:|---:|---|
| `true` | 128 (todos) | 4,07 | 36,06 s | **VACÍO** |
| `false` | 128 | 4,69 | 31,29 s | **430 caracteres** |

Y con un límite holgado (900 tokens), la diferencia es de rendimiento, no de
contenido:

| Límite | `think` | Tokens | tok/s | Caracteres |
|---:|---|---:|---:|---:|
| 900 | `false` | 586 | 5,90 | respuesta completa |
| 400 | `true` | 295 | 4,69 | **18** (se quedó corto) |

`think=true` consume **más tokens para decir menos**. El razonamiento no es
gratis: es un coste real que se paga en tiempo de CPU, y en una máquina sin GPU
ese coste se nota.

### Por qué `false` es el valor por defecto y no una excepción

El umbral depende del prompt. Un prompt complejo hará razonar más antes de
responder, así que el mismo límite que funciona con «di hola» puede vaciarse con
una pregunta de cálculo. Fijar `false` hace que el comportamiento sea
predecible en lugar de depender de cuánto razone el modelo.

`test_thinking.py` mide los dos modos y está en el repositorio, para que la
decisión se pueda reproducir y discutir.

> **El patrón de fallo es el del Día 9 otra vez:** un parámetro del cliente
> (el límite de tokens) interactuando con un parámetro del modelo (thinking por
> defecto) produce un resultado que **parece** un modelo roto. El modelo
> funcionaba; la llamada estaba mal configurada. Por eso `done_reason` se
> comprueba siempre.

---

## 4. Script de chat dual

`chat_dual.py` implementa el entregable completo: chat por consola con elección
de backend al arrancar, y envío de los 5 prompts con tabla comparativa.

### Los dos backends

Ambos exponen el mismo interfaz (`preguntar`, `listar`, `disponible`), así que
cambiar de uno a otro es cambiar una variable:

| Backend | Ruta | Protocolo | Estado |
|---|---|---|---|
| `BackendOllama` | `/api/chat` | API nativa de Ollama | **Ejecutado y medido** |
| `BackendRemoto` | `/v1/chat/completions` | Compatible con OpenAI | **Implementado, no ejecutado** |

`BackendRemoto` habla el mismo protocolo que usan vLLM, llama.cpp server, TGI y
**los despliegues de Azure**. Por eso la parte inaccesible del enunciado se
ejecuta sin tocar una línea en cuanto haya URL:

```powershell
python chat_dual.py --backend local
python chat_dual.py --backend remoto --endpoint https://<vm>:8000/v1 --api-key <clave>
python chat_dual.py --backend ambos --endpoint https://<vm>:8000/v1   # comparativa real
```

### Los 5 prompts

Cada uno ataca una capacidad distinta. Comparar backends con cinco variantes del
mismo prompt no diría nada.

| Id | Capacidad | Qué mide |
|---|---|---|
| P1 | Razonamiento aritmético | 3 importes con respuesta única y verificable |
| P2 | Generación de código | Algoritmo con caso límite (`None`) |
| P3 | Explicación | Analogía, 3 frases, sin jerga |
| P4 | Instrucción estricta | JSON válido, sin markdown ni texto extra |
| P5 | Memoria de instrucción | 3 reglas simultáneas: mayúsculas, punto y coma, prohibición |

### Verificación automática

Cada prompt tiene una comprobación programática, **anotada antes de ejecutar**:

| Prompt | Comprobación | Verdad terreno |
|---|---|---|
| P1 | ¿Aparecen 72.000, 63.000 y 45.000? | Aritmética exacta |
| P2 | ¿Bloque de código? ¿`None`? ¿`sorted`/`max`? | Forma y caso límite |
| P3 | ¿3 frases? ¿Analogía? ¿Sin términos técnicos? | Lectura de la forma |
| P4 | ¿JSON parseable con `lenguaje` y `uso`? | Formato exacto |
| P5 | ¿Mayúsculas? ¿`;` final? ¿Sin «gato»? | Las 3 reglas a la vez |

Esto mide la **forma** de la respuesta, no su corrección semántica. P3 en concreto
comprueba que hay analogía y que no hay jerga, pero **no juzga si la analogía es
buena**: eso requiere una persona, y fingir lo contrario con una regex sería
inventarse una métrica.

---

## 5. Resultados medidos

Ejecución real del 2026-09-15, `gemma4:e4b`, Q4_K_M, CPU-only, `think=false`,
temperatura 0,7.

| Prompt | Tokens | Latencia | tok/s | Verificación |
|---|---:|---:|---:|---|
| P1 cálculo | 586 | 102,32 s | 5,90 | **3/3 importes** ✅ |
| P2 código | 159 | 36,23 s | 5,31 | Bloque, `None`, `sorted` ✅ |
| P3 explicación | 91 | 33,83 s | 2,93 | 3 frases, analogía, sin jerga ✅ |
| P4 instrucción | 26 | 11,05 s | 3,20 | JSON válido, sin markdown ✅ |
| P5 multilínea | 20 | 10,58 s | 2,57 | Las 3 reglas a la vez ✅ |

**5/5 en verificación automática.** Contenido íntegro en
`resultados/comparativa_backends.md`.

### Los números que de verdad importan

**Entre 2,5 y 5,9 tokens por segundo.** En una respuesta de 586 tokens, el
usuario espera **1 minuto y 42 segundos**. En una de 20 tokens, 10,6 segundos.

Esto define qué es usable en local con esta máquina:

| Caso de uso | ¿Viable aquí? |
|---|---|
| Clasificación, extracción de campos, renombrado | ✅ Segundos |
| Resumir un párrafo | ✅ 10–40 s |
| Generar código corto | ⚠️ 30–60 s y con revisión |
| Chat conversacional fluido | ❌ 2–5 tok/s se nota como lentitud |
| Cualquier cosa por encima de 500 tokens | ❌ Se agota la paciencia |

### Un detalle sobre la latencia que engaña

P3 tardó 33,83 s para generar **91 tokens** (2,93 tok/s), mientras que P1 tardó
102,32 s para generar **586** (5,90 tok/s). La respuesta corta tiene menos
throughput.

La explicación está en `prompt_eval_duration`: procesar un prompt largo consume
tiempo antes de empezar a generar el primer token, y ese tiempo **cuenta dentro
de la latencia total** pero no en `eval_duration`. P3 tiene el prompt más corto
pero el más «pensado» (pedir una analogía sin jerga es más costoso de procesar
que una división).

Por eso los dos números se reportan por separado:

- **`tok/s`** mide la generación pura. Es la cifra del motor.
- **`latencia total`** es lo que espera el usuario. Incluye la evaluación del
  prompt.

Confundirlos es el error más fácil al medir un modelo local, y da cifras que no
reproducen.

### Coste de la primera petición

| Concepto | Valor |
|---|---|
| Carga del modelo en RAM | **34,54 s** (primera vez) |
| Carga con el modelo ya resident | **0,03 s** |
| Tiempo de descarga | 6,6 GB (~13 min a velocidad variable) |

Los 34,54 s son el precio de la primera petición. Después es 0,03 s. Es la razón
por la que **no se puede medir un modelo recién descargado sin hacer una
petición de calentamiento**: la primera incluye la carga desde disco.

---

## 6. Local vs. nube: el razonamiento

La comparación **no se ha ejecutado** (no hay VM de Azure). Lo que sí se puede
hacer es razonar sobre los datos medidos y sobre lo que la arquitectura implica,
marcando qué es medida y qué es deducción.

### La tabla que el enunciado pedía, con la parte remota vacía

| Métrica | Local (medido) | Remoto (Azure) |
|---|---:|---|
| Modelos disponibles | 1–2, los que quepan en RAM | Cientos, sin límite de memoria |
| Parámetros máximos | 31B (con cuantización y RAM) | Sin límite práctico |
| tok/s | **2,5–5,9** (medido) | 20–80 (típico en GPU) |
| Latencia | 10–102 s (medido) | 0,5–3 s |
| Coste | 0 € tras el hardware | Por token |
| Privacidad | **El dato no sale del equipo** | Sale a un servidor |
| Requiere red | No | Sí |
| Multiusuario | No, un equipo | Sí |

### Cuándo usar cada uno

| Situación | Decisión | Motivo |
|---|---|---|
| Datos sensibles (salud, clientes, industrial) | **Local** | El dato no sale. Es la única ventaja que no se compra |
| Coste fijo alto, muchos usuarios | Remoto | Pagar por uso sale mejor que un equipo con GPU |
| GPU dedicada en el servidor | Remoto | Aquí no hay GPU, y eso no se arregla con software |
| Prototipo sin GPU | **Local** | 15,7 GB de RAM permiten 4,5B cuantizados |
| Chat conversacional | Remoto | 2,5 tok/s no es un chat |
| Extracción / clasificación en lote | **Local** | 2–10 s por elemento, sin coste marginal |
| Trabajar sin conexión | **Local** | Sin VPN ni red |
| Reproducibilidad exacta | **Local** | Mismos pesos, mismo hardware, mismo resultado |

### La ventaja que no se compra

La más importante de la columna local no es el coste, es que **el dato no sale
del equipo**. Los días anteriores de esta práctica hubo que rotar API keys de
Gemini que quedaron expuestas en el chat. Un modelo local no tiene ese problema
porque **no hay clave que filtrar**: no hay nada que robar más allá del portátil.

Eso tiene un coste que no es monetario: el hardware es tuyo y hay que
mantenerlo. Un modelo cuantizado en local es más rápido que depender de una red
corporativa, pero es más lento que un modelo grande en la nube. **No sustituye,
complementa.**

### La limitación honesta de este equipo

Sin GPU dedicada, el modelo de 4,5 B **duerme en RAM y calcula en CPU**. Con
una GPU de 12 GB, `gemma4:12b` daría entre 20 y 40 tok/s: unas 8 veces más. La
arquitectura del modelo es la misma; lo que cambia es dónde ejecutan las
multiplicaciones de matrices.

---

## 7. Lo que no se puede concluir

| Afirmación | Por qué **no** |
|---|---|
| «Local es más rápido que Azure» | **Sin medir. Y probablemente falso:** una GPU de servidor da 20–80 tok/s, este equipo 2,5–5,9. La conclusión honesta es la contraria: aquí local es **más lento** y el tradeoff es privacidad y coste, no velocidad. |
| «Azure da mejores respuestas» | Sin medir y, por tamaño de modelo, casi con certeza sí: 405B o GPT-4 contra 4,5 B. Un modelo 90 veces mayor gana. No hace falta ni ejecutarlo para saberlo, pero tampoco se puede **cuantificar** la diferencia sin medirla. |
| «La calidad es equivalente en local y remoto» | No. Se puede definir la calidad del mismo modo en ambos, y sería el experimento correcto. No está hecho. |
| «5/5 en verificación significa que Gemma 4 E4B es fiable» | **5 prompts no son una evaluación.** Es una muestra mínima. La verificación automática mide forma, no corrección semántica. En el Día 9 ya se cayó en esto: 5/5 en un corpus pequeño no autoriza a afirmar nada general. |
| «Gemma 4 E4B es mejor que Llama 3.1 8B» | No se compararon entre sí en este día. Both Q4_K_M, ambos locales, pero **no se ejecutó el mismo prompt en ambos** con métricas. |
| «El modo thinking mejora la calidad» | No medido. Solo se midió su **coste**. Que razone más no implica que acierte más, y en CPU ese coste es una latencia que el usuario paga. |
| «E4B necesita 6,6 GB» | Necesita **6,6 GB de descarga**, pero en RAM el consumo es mayor: LLM + proyector (478 M parámetros) + KV cache. Los números no son la misma cosa. |

**Lo que sí se puede concluir:**

1. Gemma 4 E4B **funciona desplegado en local** por CLI y por API REST, y
   responde correctamente en las 5 capacidades probadas.
2. El coste real es de **2,5 a 5,9 tok/s en CPU sin GPU**. Con eso se decide qué
   casos de uso son viables (§5).
3. El **modo thinking viene activado por defecto** y con límites bajos de tokens
   produce respuestas vacías. Hay que enviar `think=false`.
4. `load_duration` baja de **34,54 s a 0,03 s** tras la primera petición: hace
   falta calentamiento antes de medir.
5. La latencia total y el tok/s **miden cosas distintas**, y confundirlos da
   cifras irreproducibles.
6. En este equipo, **la carga de 34,54 s es el coste de arranque y la RAM el
   límite**; la GPU es lo que falta.

---

## 8. Errores encontrados

Cinco, y cuatro son del mismo tipo: **algo funcionó a medias y dio la impresión de
que el modelo estaba roto**.

| # | Error | Síntoma | Causa | Solución |
|---|---|---|---|---|
| 1 | Modo thinking activo | Respuesta vacía con `num_predict` bajo | Thinking por defecto come el presupuesto | `think=false` en el cliente, con medición en `test_thinking.py` |
| 2 | `--comparar` no ejecutaba nada | Terminaba con **código 0** tras imprimir el inventario | `--backend` valía `listar` por defecto y cortaba el flujo antes | `--comparar` se evalúa antes. Un fallo silencioso con código de éxito es el peor tipo de fallo |
| 3 | `UnboundLocalError: re` | Crash en la verificación de P1 | `import re` dentro de una función lo convertía en local, pero se usaba antes | `import re` a nivel de módulo |
| 4 | `NameError: escolhas` | Crash en el cliente remoto | `escolhas` en la asignación, `Elecciones` en el uso — desajuste de mayúsculas | Renombrado. **Solo apareció al probar el segundo transporte**, que es justo para lo que existen dos clientes |
| 5 | P1 falló 2/3 importes | Verificaciónarna 2 de 3 | **Truncamiento a 400 tokens**, no error del modelo | Repetido con 900 tokens → 3/3 |

### El más instructive: el nº 5

P1 falló 2 de 3 comprobaciones y la lectura natural era «el modelo se equivoca
en uno de los tres socios». Al subir el límite a 900 tokens el resultado fue
**3/3**, con la respuesta completa y una tabla de verificación que el propio
modelo añadió (`72.000 + 63.000 + 45.000 = 180.000`).

El modelo no se equivocaba: la llamada estaba truncando su respuesta. Un modelo
cuto que da un número parcial es **indistinguible** de un modelo que calcula mal,
y solo se distingue mirando `done_reason` y pidiendo más tokens.

> **Los errores nº 3, 4 y 5 tienen la misma raíz que los del Día 9:** desajustes
> de una línea entre un nombre y su uso, casi siempre en la frontera
> español/inglés. Un linter no los detecta porque el código es sintácticamente
> válido. Se detectan ejecutando, que es lo único que revela el desajuste.

---

## 9. Reproducir

```powershell
# Requisitos: Python 3.10+ y Ollama (https://ollama.com)
ollama --version
ollama pull gemma4:e4b          # 6,6 GB, ~13 min

# Ver el inventario y qué responde cada backend
python chat_dual.py --backend listar

# Chat por consola con el backend local
python chat_dual.py --backend local --modelo gemma4:e4b
# escribe 'salir' para terminar, 'info' para ver el estado

# Los 5 prompts con tabla comparativa
python chat_dual.py --backend local --comparar --modelo gemma4:e4b --max-tokens 900

# Guardar el informe
python chat_dual.py --backend local --comparar --modelo gemma4:e4b --max-tokens 900 --guardar resultados/

# Con el modo thinking activado (ver §3)
python chat_dual.py --backend local --comparar --think

# Medir think=true vs think=false
python test_thinking.py --tokens 128

# Cuando haya VM de Azure, la comparación real sin tocar código
python chat_dual.py --backend ambos --endpoint https://<vm>:8000/v1 --api-key <clave> --comparar
```

Opciones:

```
--backend local|remoto|ambos|listar   Backend a usar
--host HOST                           Host de Ollama (por defecto localhost:11434)
--endpoint URL                        URL del backend remoto compatible con OpenAI
--api-key CLAVE                       Token Bearer del backend remoto
--modelo ID                           Modelo a usar
--comparar                            Envia los 5 prompts y genera tabla
--reps N                              Pasadas por prompt; se reporta la mediana
--think                               Activa el modo thinking (por defecto OFF)
--temperatura F / --max-tokens N      Parametros de generacion
--guardar DIR                         Escribe el informe Markdown en DIR
```

### Una advertencia práctica sobre `--max-tokens`

**No pongas el límite por debajo de 500.** Con 400 tokens, P1 se trunca y la
verificación falla aunque el modelo acierte. El valor por defecto de
`chat_dual.py` cuando se compara debería ser holgado; si se va justo, el prompt
sale incompleto y parece un error del modelo.

---

## 10. Fuentes

**Documentación y fichas oficiales:**

- Ollama, descarga e instalación — <https://ollama.com/download>
- Ollama, API REST — <https://github.com/ollama/ollama/blob/main/docs/api.md>
- Ollama, tags de Gemma 4 — <https://ollama.com/library/gemma4/tags>
- Gemma 4 model card (capacidades, thinking, projector) —
  <https://huggingface.co/google/gemma-4-E4B-it>
- Gemma 4 overview y tabla de memoria — <https://ai.google.dev/gemma/docs/core>
- Gemma 4 Technical Report (arXiv 2607.02770) — <https://arxiv.org/html/2607.02770v1>

**Recursos del enunciado:**

- NetworkChuck, *Host ALL your AI Locally* (EN) —
  <https://www.youtube.com/watch?v=Wjrdr0NU4Sk>
- Dot CSV, *¡EMPIEZA A USAR la IA GRATIS en tu PC!* (ES) —
  <https://www.youtube.com/watch?v=Ye7cERMKZDY>

**Datos locales:** `referencias_ollama.json` (tags verificados el 2026-09-15),
`resultados/comparativa_backends.md` (respuestas y métricas de la ejecución).

# Dia_6-10-2026
