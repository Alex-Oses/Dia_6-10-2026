# Local vs. remoto â€” DÃ­a 10 (2026-10-05 11:48:08)

Temperatura 0.7, mÃ¡x. 900 tokens, 1 pasada(s) por prompt. Las cifras son medianas cuando reps > 1.

## Metricas

| Prompt | Backend | Tokens | Latencia | tok/s | Estabilidad |
|---|---|---:|---:|---:|---|
| P1_calculo | local (Ollama) | 586 | 102.32 s | 5.9 | estable |
| P2_codigo | local (Ollama) | 159 | 36.23 s | 5.31 | estable |
| P3_explicacion | local (Ollama) | 91 | 33.83 s | 2.93 | estable |
| P4_instruccion | local (Ollama) | 26 | 11.05 s | 3.2 | estable |
| P5_multilinea | local (Ollama) | 20 | 10.58 s | 2.57 | estable |

## Verificacion automatica

| Prompt | Backend | Comprobacion |
|---|---|---|
| P1_calculo | local (Ollama) | 3/3 importes correctos |
| P2_codigo | local (Ollama) | {"bloque_codigo": true, "solo_codigo": true, "menciona_none": true, "usa_sorted_o_max": true, "lineas": 15} |
| P3_explicacion | local (Ollama) | {"frases": 4, "aprox_3": true, "tiene_analogia": true, "evita_tecnico": true} |
| P4_instruccion | local (Ollama) | JSON valido=True Â· sin markdown=True |
| P5_multilinea | local (Ollama) | mayusculas=True Â· punto y coma=True Â· prohibicion=True |

## Respuestas completas

### P1_calculo â€” local (Ollama)

```text
Aquí tienes el cálculo detallado para repartir el beneficio:

**Datos:**

* **Beneficio total a repartir:** $180.000 \text{ EUR}$
* **Participación Socio 1:** $40\%$
* **Participación Socio 2:** $35\%$
* **Participación Socio 3:** $25\%$

---

### Cálculo para cada socio

Para saber cuánto le toca a cada socio, multiplicamos el beneficio total por el porcentaje de participación de cada uno.

**1. Socio 1 (40%):**

$$\text{Monto Socio 1} = \text{Beneficio Total} \times \text{Participación Socio 1}$$
$$\text{Monto Socio 1} = 180.000 \text{ EUR} \times 0,40$$
$$\text{Monto Socio 1} = 72.000 \text{ EUR}$$

**2. Socio 2 (35%):**

$$\text{Monto Socio 2} = \text{Beneficio Total} \times \text{Participación Socio 2}$$
$$\text{Monto Socio 2} = 180.000 \text{ EUR} \times 0,35$$
$$\text{Monto Socio 2} = 63.000 \text{ EUR}$$

**3. Socio 3 (25%):**

$$\text{Monto Socio 3} = \text{Beneficio Total} \times \text{Participación Socio 3}$$
$$\text{Monto Socio 3} = 180.000 \text{ EUR} \times 0,25$$
$$\text{Monto Socio 3} = 45.000 \text{ EUR}$$

---

### Verificación (Comprobación)

Para asegurarnos de que la distribución es correcta, sumamos las partes repartidas y deben sumar el beneficio total:

$$72.000 \text{ EUR} + 63.000 \text{ EUR} + 45.000 \text{ EUR} = 180.000 \text{ EUR}$$

---

### Respuesta Final

A cada socio le toca lo siguiente:

* **Socio 1 (40%):** **$72.000 \text{ EUR}$**
* **Socio 2 (35%):** **$63.000 \text{ EUR}$**
* **Socio 3 (25%):** **$45.000 \text{ EUR}$**
```

### P2_codigo â€” local (Ollama)

```text
```python
def segundo_mayor_distinto(lista):
    """
    Recibe una lista de enteros y devuelve el segundo mayor valor distinto.
    Si no hay segundo mayor, devuelve None.
    """
    if not lista:
        return None

    # Obtener valores únicos y ordenarlos de forma descendente
    valores_unicos_ordenados = sorted(list(set(lista)), reverse=True)

    # Verificar si hay al menos dos valores distintos
    if len(valores_unicos_ordenados) < 2:
        return None
    else:
        # El segundo elemento en la lista ordenada es el segundo mayor distinto
        return valores_unicos_ordenados[1]
```
```

### P3_explicacion â€” local (Ollama)

```text
Imagina que tienes una radio que reproduce música con muchísimos matices de volumen. La cuantización es como si le dijeras a esa radio que solo debe usar unos pocos niveles de volumen fijos, en lugar de infinitos. Así, el modelo se vuelve más pequeño y rápido para funcionar, aunque pierda un poco de la sutileza original. Es como simplificar un dibujo detallado a un boceto más básico pero funcional.
```

### P4_instruccion â€” local (Ollama)

```text
{"lenguaje": "Marinero/Local", "uso": "Navegación, pesca, turismo costero"}
```

### P5_multilinea â€” local (Ollama)

```text
LAS GOTAS DE AGUA CAEN DEL CIELO CON UN RITMO SUAVE;
```

## Conclusion

**La comparacion local vs. remoto NO se ha completado.** La VM de Azure de Inetum no ha estado accesible desde este equipo, asi que solo hay mediciones del backend local. Los numeros remotos faltan y no se inventan.

**Lo que si se puede afirmar con lo medido:** el coste en tokens/segundo y en latencia del despliegue local, que es la cifra contra la que se decide si un caso de uso aguanta en local.

**Lo que no:** que un backend sea mejor que otro. Con un solo backend activo no hay comparacion; hacen falta los dos medidos con el mismo prompt, las mismas pasadas y una verdad terreno anotada.
