# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y Flask
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de dependencias
COPY requirements.txt .

# Actualizar pip antes de instalar dependencias
RUN pip install --no-cache-dir --upgrade pip

# Instalar las dependencias especificadas en requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Gunicorn
RUN pip install gunicorn

# Copiar el resto de la aplicación al contenedor
COPY . .

# Exponer el puerto en el que Flask correrá
EXPOSE 5000

# Ejecutar la aplicación usando Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
