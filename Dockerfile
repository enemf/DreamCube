FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Установка основных Python пакетов
RUN pip install --upgrade pip setuptools wheel

# Копирование локального кода
WORKDIR /workspace
COPY . .

# Установка pytorch3d перед остальными зависимостями
RUN pip install fvcore iopath
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

# Установка зависимостей из requirements.txt
RUN pip install -r requirements.txt

# Предварительная загрузка модели из HuggingFace
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('KevinHuang/DreamCube')"

# Создание директории для выходных файлов
RUN mkdir -p /workspace/outputs

# Установка переменных окружения
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0
ENV GRADIO_ALLOW_REMOTE=true

# Экспонирование порта для Gradio
EXPOSE 7422

# Команда запуска
CMD ["python", "app.py", "--use-gradio"] 