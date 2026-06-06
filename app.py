import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# CORRECCIÓN DE IMPORTACIONES PARA LANGCHAIN v1.0+
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Importamos la función separada del módulo local
from rag_pipeline import inicializar_pipeline_rag
    
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# Cargar las variables de entorno del archivo .env
load_dotenv()

# VALIDACIÓN DE ARQUITECTURA
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ Error: La variable OPENAI_API_KEY no está configurada en el archivo .env")

MODELO_LLM = "gpt-4o-mini"

def ejecutar_aplicacion():
    # Inicializar el motor RAG importado
    vector_store = inicializar_pipeline_rag()
    if not vector_store:
        return

    # Configurar el recuperador (Retriever)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Configurar el Modelo de Lenguaje
    llm = ChatOpenAI(model=MODELO_LLM, temperature=0)

    # Definir la estructura del Prompt del sistema
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "Eres un asistente virtual experto encargado de responder preguntas basándote "
            "únicamente en el contexto proporcionado. Si no sabes la respuesta o no está "
            "en los documentos, di explícitamente que no posees esa información.\n\n"
            "CONTEXTO DE LOS DOCUMENTOS LOCALES:\n{context}"
        )),
        ("human", "{input}"),
    ])

    # Ensamblar la cadena de ejecución RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n🚀 ¡Sistema RAG listo! Puedes empezar a chatear con tus documentos.")
    print("Para salir del chat, escribe 'salir' o 'exit'.\n")
    
    while True:
        pregunta = input("👤 Tu pregunta: ")
        if pregunta.lower() in ['salir', 'exit']:
            print("👋 Cerrando el asistente RAG. ¡Hasta luego!")
            break
            
        if not pregunta.strip():
            continue

        print("🤖 Pensando...")
        try:
            respuesta = rag_chain.invoke({"input": pregunta})
            print("\n================ RESPUESTA ================")
            print(respuesta["respuesta"])
            print("===========================================\n")
        except Exception as e:
            print(f"❌ Ocurrió un error al consultar OpenAI: {e}\n")

if __name__ == "__main__":
    ejecutar_aplicacion()