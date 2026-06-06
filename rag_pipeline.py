import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# CONFIGURACIÓN INTERNA DEL PIPELINE
RUTA_LOCAL_PDFS = "./documentos_locales"
MODELO_EMBEDDING = "text-embedding-3-small"

def inicializar_pipeline_rag():
    """
    Ejecuta las 4 fases principales del flujo RAG para procesar PDFs locales.
    Retorna el vector store listo para realizar búsquedas.
    """
    # Asegurar que la carpeta local exista
    Path(RUTA_LOCAL_PDFS).mkdir(parents=True, exist_ok=True)
    
    # Verificar si existen archivos PDF
    pdf_files = list(Path(RUTA_LOCAL_PDFS).glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  La carpeta '{RUTA_LOCAL_PDFS}' está vacía.")
        print("👉 Por favor, guarda tus archivos PDF en ella antes de continuar.")
        return None

    print(f"📂 Detectados {len(pdf_files)} archivos PDF para procesar.")

    # ==========================================
    # FASE 1: EXTRACCIÓN (Extraction)
    # ==========================================
    print("\n[1/4] Extrayendo contenido de los PDFs...")
    loader = PyPDFDirectoryLoader(RUTA_LOCAL_PDFS)
    documentos_extraidos = loader.load()
    print(f"✅ Extracción completada. Total de páginas leídas: {len(documentos_extraidos)}")

    # ==========================================
    # FASE 2: FRAGMENTACIÓN (Chunking)
    # ==========================================
    print("\n[2/4] Fragmentando el texto (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    fragmentos = text_splitter.split_documents(documentos_extraidos)
    print(f"✅ Fragmentación completada. Total de chunks generados: {len(fragmentos)}")

    # ==========================================
    # FASE 3: INCRUSTACIÓN (Embedding) & 
    # FASE 4: ALMACENAMIENTO Y BÚSQUEDA (Storage)
    # ==========================================
    print("\n[3/4 y 4/4] Generando vectores y guardando en ChromaDB...")
    embeddings = OpenAIEmbeddings(model=MODELO_EMBEDDING)
    
    # Inicialización del almacenamiento vectorial en memoria usando langchain-chroma
    print("🔹 Indexando fragmentos en ChromaDB (Motor de producción)...")
    vector_store = Chroma.from_documents(
        documents=fragmentos, 
        embedding=embeddings
    )
    print("✅ ChromaDB cargado correctamente en memoria.")

    print("✅ Vectores almacenados exitosamente en la base de datos local ChromaDB.")
    return vector_store