FROM local-torch-geometric
WORKDIR /app
COPY . .
ENV PYTHONPATH /app
RUN ls
RUN pip install -r requirements.txt
CMD ["python3"]
