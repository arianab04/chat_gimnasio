"""
rag.py — Pre-entrega 3: Sistema de recuperación semántica local (RAG)
=====================================================================

Flujo End-to-End de RAG sobre la base vectorial creada por `ingest.py`.

    pregunta -> embedding -> búsqueda en ChromaDB -> prompt con contexto
             -> LLM (async) -> PydanticOutputParser -> RespuestaRAG

Uso:
    python rag.py                 # corre las dos pruebas obligatorias
    python rag.py --interactivo   # modo pregunta/respuesta por consola

Requisito previo:
    python ingest.py              # crea la carpeta ./vectorstore
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from operator import itemgetter
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------------------------------------------------------------------
# PASO 0 — Configuración
# ---------------------------------------------------------------------------

load_dotenv()

VECTORSTORE_PATH = "vectorstore"

# ¡IMPORTANTE! Debe ser EXACTAMENTE el mismo modelo usado en ingest.py.
# Si indexás con un modelo y consultás con otro, las distancias vectoriales
# no tienen sentido y los resultados son aleatorios (error #1 de la consigna).
EMBEDDING_MODEL = "text-embedding-3-small"

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# top_k entre 3 y 5: evita el "contexto infinito" / Lost in the Middle.
TOP_K = 4


def _validar_entorno() -> None:
    """Falla temprano y con un mensaje claro si falta algo."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "ERROR: falta OPENAI_API_KEY.\n"
            "Creá un archivo .env en la raíz del proyecto con:\n"
            "    OPENAI_API_KEY=sk-...\n"
        )
    if not Path(VECTORSTORE_PATH).exists():
        sys.exit(
            f"ERROR: no existe la carpeta '{VECTORSTORE_PATH}'.\n"
            "Ejecutá primero:  python ingest.py\n"
        )


# ---------------------------------------------------------------------------
# PASO 1 — Esquemas Pydantic (contrato de salida)
# ---------------------------------------------------------------------------


class RespuestaLLM(BaseModel):
    """Lo que el modelo de lenguaje DEBE devolver, en JSON.

    Este es el esquema que consume el PydanticOutputParser: a partir de él
    se generan las instrucciones de formato que se inyectan en el prompt.
    """

    respuesta: str = Field(
        description=(
            "Respuesta redactada en español, basada únicamente en el CONTEXTO. "
            "Si el contexto no alcanza, el texto debe decir explícitamente que "
            "no se dispone de esa información."
        )
    )
    fuentes_citadas: list[str] = Field(
        default_factory=list,
        description=(
            "Nombres de los archivos del CONTEXTO efectivamente usados para "
            "responder. Lista vacía si no se usó ninguno."
        ),
    )
    contexto_suficiente: bool = Field(
        description="True si el CONTEXTO alcanzaba para responder; False si no."
    )


class RespuestaRAG(BaseModel):
    """Objeto final que devuelve `get_rag_response`.

    Combina lo que dijo el LLM con la trazabilidad real de la recuperación
    (qué archivos se recuperaron y cuántos fragmentos entraron al prompt).
    """

    pregunta: str
    respuesta: str
    fuentes: list[str] = Field(
        description="Archivos recuperados desde ChromaDB (metadata real)."
    )
    fuentes_citadas: list[str] = Field(
        description="Archivos que el LLM declara haber usado."
    )
    contexto_suficiente: bool
    fragmentos_recuperados: int


# ---------------------------------------------------------------------------
# PASO 2 — Capa de recuperación (Retriever sobre ChromaDB)
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings,
)

# El retriever convierte la pregunta en embedding y trae los TOP_K fragmentos
# más cercanos. Es un Runnable: se puede encadenar con | dentro de LCEL.
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})


def formatear_contexto(docs: list[Document]) -> str:
    """Convierte los Documents recuperados en un bloque de texto para el prompt.

    Numerar y etiquetar cada fragmento con su archivo de origen es lo que
    después le permite al LLM citar fuentes con precisión.
    """
    if not docs:
        return "(No se recuperó ningún fragmento.)"

    bloques = []
    for i, doc in enumerate(docs, start=1):
        origen = doc.metadata.get("source", "desconocido")
        bloques.append(f"[Fragmento {i} | fuente: {origen}]\n{doc.page_content}")
    return "\n\n".join(bloques)


def _nombres_de_fuentes(docs: list[Document]) -> list[str]:
    """Lista de archivos únicos recuperados, sin repetir y en orden estable."""
    vistos: list[str] = []
    for doc in docs:
        origen = doc.metadata.get("source", "desconocido")
        if origen not in vistos:
            vistos.append(origen)
    return vistos


# ---------------------------------------------------------------------------
# PASO 3 — Prompt de sistema ("filtro de veracidad") + parser
# ---------------------------------------------------------------------------

parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

SYSTEM_PROMPT = """Sos un asistente informativo del gimnasio FitLife.

REGLAS ESTRICTAS:
1. Respondé ÚNICAMENTE con la información del CONTEXTO que se te entrega.
2. No uses conocimiento previo ni inventes datos, precios, horarios ni políticas.
3. Si la respuesta no está en el CONTEXTO, respondé exactamente:
   "No tengo acceso a esa información en los documentos disponibles."
   y marcá contexto_suficiente en false.
4. No mezcles información de fragmentos que hablen de otro tema.
5. Citá en fuentes_citadas solo los archivos que realmente usaste.

FORMATO DE SALIDA (obligatorio):
{format_instructions}

Devolvé solo el JSON, sin texto adicional ni bloques de código."""

HUMAN_PROMPT = """CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
).partial(format_instructions=parser.get_format_instructions())


# ---------------------------------------------------------------------------
# PASO 4 — Modelo de lenguaje
# ---------------------------------------------------------------------------

# temperature=0 -> respuestas deterministas y menos propensas a alucinar.
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# PASO 5 — Cadena LCEL completa
# ---------------------------------------------------------------------------


def _armar_respuesta_final(datos: dict) -> RespuestaRAG:
    """Último eslabón: fusiona la salida del LLM con los datos de recuperación."""
    salida: RespuestaLLM = datos["salida_llm"]
    docs: list[Document] = datos["docs"]

    return RespuestaRAG(
        pregunta=datos["pregunta"],
        respuesta=salida.respuesta,
        fuentes=_nombres_de_fuentes(docs),
        fuentes_citadas=salida.fuentes_citadas,
        contexto_suficiente=salida.contexto_suficiente,
        fragmentos_recuperados=len(docs),
    )


# Sub-cadena de generación: prompt -> LLM -> parser Pydantic.
cadena_generacion = prompt | llm | parser

# Cadena completa:
#   1) recupera documentos y conserva la pregunta,
#   2) arma el string de contexto,
#   3) genera y parsea la respuesta,
#   4) ensambla el objeto RespuestaRAG.
cadena_rag = (
    RunnableParallel(
        pregunta=itemgetter("pregunta"),
        docs=itemgetter("pregunta") | retriever,
    )
    | RunnablePassthrough.assign(contexto=lambda x: formatear_contexto(x["docs"]))
    | RunnablePassthrough.assign(salida_llm=cadena_generacion)
    | RunnableLambda(_armar_respuesta_final)
)


# ---------------------------------------------------------------------------
# PASO 6 — Interfaz pública asíncrona
# ---------------------------------------------------------------------------


async def get_rag_response(query: str) -> RespuestaRAG:
    """Ejecuta el flujo RAG completo de forma asíncrona.

    Args:
        query: la pregunta del usuario en lenguaje natural.

    Returns:
        RespuestaRAG con el texto, las fuentes y la cantidad de fragmentos.
    """
    try:
        return await cadena_rag.ainvoke({"pregunta": query})
    except OutputParserException:
        # Red de seguridad: si el LLM devolvió un JSON inválido, no rompemos
        # el programa; devolvemos una respuesta honesta y trazable.
        docs = await retriever.ainvoke(query)
        return RespuestaRAG(
            pregunta=query,
            respuesta=(
                "No pude generar una respuesta con formato válido. "
                "Volvé a intentar la consulta."
            ),
            fuentes=_nombres_de_fuentes(docs),
            fuentes_citadas=[],
            contexto_suficiente=False,
            fragmentos_recuperados=len(docs),
        )


# ---------------------------------------------------------------------------
# PASO 7 — Pruebas obligatorias y modo interactivo
# ---------------------------------------------------------------------------

PREGUNTA_OK = "¿Qué incluye el Plan Premium?"
PREGUNTA_TRAMPA = "¿Cuánto cuesta la membresía anual en dólares?"


def _imprimir(titulo: str, r: RespuestaRAG) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    print(f"PREGUNTA:              {r.pregunta}")
    print(f"RESPUESTA:             {r.respuesta}")
    print(f"FUENTES RECUPERADAS:   {r.fuentes}")
    print(f"FUENTES CITADAS:       {r.fuentes_citadas}")
    print(f"CONTEXTO SUFICIENTE:   {r.contexto_suficiente}")
    print(f"FRAGMENTOS RECUPERADOS:{r.fragmentos_recuperados}")


async def ejecutar_pruebas() -> None:
    """Prueba 1: respuesta que SÍ está en los documentos.
    Prueba 2: 'pregunta trampa' que NO está -> el modelo no debe alucinar."""
    ok, trampa = await asyncio.gather(
        get_rag_response(PREGUNTA_OK),
        get_rag_response(PREGUNTA_TRAMPA),
    )
    _imprimir("PRUEBA 1 — Pregunta respondible", ok)
    _imprimir("PRUEBA 2 — Pregunta trampa (no debe alucinar)", trampa)
    print()


async def modo_interactivo() -> None:
    print("Modo interactivo. Escribí 'salir' para terminar.\n")
    while True:
        pregunta = input("Vos> ").strip()
        if pregunta.lower() in {"salir", "exit", "quit", ""}:
            print("Hasta luego.")
            break
        r = await get_rag_response(pregunta)
        print(f"\nDilsy> {r.respuesta}")
        print(f"       fuentes: {r.fuentes_citadas or r.fuentes}\n")


def main() -> None:
    argumentos = argparse.ArgumentParser(description="RAG local sobre ChromaDB")
    argumentos.add_argument(
        "--interactivo",
        action="store_true",
        help="Preguntar por consola en lugar de correr las pruebas.",
    )
    args = argumentos.parse_args()

    _validar_entorno()

    if args.interactivo:
        asyncio.run(modo_interactivo())
    else:
        asyncio.run(ejecutar_pruebas())


if __name__ == "__main__":
    main()
