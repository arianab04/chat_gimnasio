# Pre-entrega 3 — Sistema de recuperación semántica local (RAG)

Asistente que responde preguntas sobre el gimnasio **FitLife** usando
únicamente la documentación interna, con ChromaDB como base vectorial local,
LangChain (LCEL) para la orquestación y un `PydanticOutputParser` para
garantizar una salida estructurada.

## Arquitectura

```
data/*.txt ──► ingest.py ──► embeddings ──► ChromaDB (./vectorstore)
                                                  │
pregunta ──► embedding ──► búsqueda top_k ────────┘
          └──► prompt (contexto + reglas) ──► LLM (async) ──► PydanticOutputParser ──► RespuestaRAG
```

| Archivo | Rol |
|---|---|
| `ingest.py` | Módulo de ingesta: carga, chunking (500 tokens / 50 overlap) y persistencia en ChromaDB |
| `rag.py` | Retriever + cadena LCEL asíncrona `get_rag_response(query)` |
| `data/` | Dataset de ejemplo (planes, horarios, clases, reglamento) |
| `vectorstore/` | Base vectorial persistida (generada, no versionada) |

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Crear el archivo `.env` a partir de `.env.example`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Ejecución

```bash
python ingest.py            # 1. crea la base vectorial (si no existe)
python ingest.py --rebuild  # 1b. la reconstruye desde cero
python rag.py               # 2. corre las dos pruebas obligatorias
python rag.py --interactivo # 3. modo pregunta/respuesta por consola
```

## Decisiones técnicas

- **Chunking:** `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, 500 tokens con 50 de overlap, para que el corte se mida en tokens reales y no en caracteres.
- **`top_k = 4`:** dentro del rango 3–5 recomendado; evita el "contexto infinito" y la degradación por *Lost in the Middle*.
- **Mismo modelo de embeddings en ingesta y consulta** (`text-embedding-3-small`): usar modelos distintos hace que las distancias vectoriales no tengan sentido.
- **Persistencia:** `ingest.py` verifica si `./vectorstore` ya existe antes de re-indexar.
- **`temperature=0`** y prompt de sistema con "filtro de veracidad": el modelo responde solo con el CONTEXTO y declara explícitamente cuando no tiene la información.
- **Salida tipada:** `RespuestaRAG` incluye el texto, las fuentes recuperadas (metadata real), las fuentes que el LLM dice haber citado, un flag `contexto_suficiente` y la cantidad de fragmentos usados.

## Pruebas

1. **Pregunta respondible:** *¿Qué incluye el Plan Premium?* → responde con datos de `planes.txt` y `contexto_suficiente = True`.
2. **Pregunta trampa:** *¿Cuánto cuesta la membresía anual en dólares?* → los documentos no tienen precios, por lo que debe responder que no dispone de esa información y devolver `contexto_suficiente = False`.

## Seguridad

La clave de API se lee desde variables de entorno (`python-dotenv`). El archivo
`.env` y la carpeta `vectorstore/` están excluidos en `.gitignore`.
