FROM python:3.12
WORKDIR /appvolt
COPY requirements.txt .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8000"]