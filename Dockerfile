FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install CPU-only torch; avoding 2.5GB CUDA torch build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["python", "-u", "flask_app.py"]