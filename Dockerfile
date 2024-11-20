# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios (app.py y requisitos)
COPY . /app

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app.py"]