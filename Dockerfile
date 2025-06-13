FROM python:3.11-slim

WORKDIR /app

COPY rag_chatbot/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "rag_chatbot/src/streamlit_app.py"]
