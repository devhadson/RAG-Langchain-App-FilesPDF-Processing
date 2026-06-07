## Guía de Instalación y Ejecución del Programa

Siga estos pasos estructurados paso a paso para desplegar la aplicación en su máquina local:

### Paso 1: Clonar o Crear el Directorio del Proyecto

Abra una terminal o línea de comandos y ejecute los siguientes comandos para preparar el entorno de archivos:


```bash
# Crear la carpeta del proyecto
mkdir pipeline-rag-clinico
cd pipeline-rag-clinico

# Crear la estructura de directorios requerida
mkdir documentos_locales
```

### Paso 2: Configurar el Entorno Virtual de Python

Se requiere Python 3.10 o una versión superior para asegurar la compatibilidad con los paquetes de LangChain v1.0+.

```bash
# Crear el entorno virtual llamado 'venv'
python3 -m venv venv

# Activar el entorno virtual
# En macOS / Linux:
source venv/bin/activate

# En Windows (Símbolo de sistema):
venv\Scripts\activate.bat

# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1

```

### Paso 3: Instalación de Dependencias

Una vez activado el entorno virtual (verá el indicador `(venv)` al inicio de su línea de comandos), instale los paquetes necesarios:

```bash
# Actualizar el gestor de paquetes pip
pip install --upgrade pip

# Instalar el stack tecnológico completo
pip install langchain langchain-community langchain-core langchain-openai langchain-chroma pypdf python-dotenv

```

### Paso 4: Creación de los Archivos de Código Fuente

Cree dos archivos de texto en la raíz del proyecto (`pipeline-rag-clinico/`) utilizando su editor de código favorito (VS Code, Notepad++, etc.) y guarde los códigos proporcionado en este repositorio:

1. Nombre al primero: **`rag_pipeline.py`**
2. Nombre al segundo: **`app.py`**

### Paso 5: Configurar las Variables de Entorno (Credenciales de OpenAI)

Cree un archivo llamado **`.env`** (sin nombre, solo con la extensión) en la raíz del directorio del proyecto:

```text
# Contenido del archivo .env
OPENAI_API_KEY=tu_clave_secreta_real_aquí_sk_...

```

### Paso 6: Alimentar la Base de Datos y Ejecutar el Sistema

1. Busque sus archivos PDF médicos (incluidos los reportes que contienen la información de la **H.C. 81743**).
2. Copie y pegue estos archivos PDF dentro de la carpeta `./documentos_locales/`.
3. Lance la aplicación ejecutando el script principal desde su terminal activa:

```bash
python app.py
```

### Paso 7: Interactuar con el Sistema

* Al iniciar, el programa leerá los PDF de la carpeta local y mostrará en pantalla el conteo de páginas y fragmentos indexados en ChromaDB.
* Cuando aparezca el prompt `👤 Tu pregunta: `, introduzca cualquiera de los **casos validados** detallados en la Sección 4.
* Para finalizar la sesión del asistente de manera limpia y segura, escriba `salir` o `exit`.