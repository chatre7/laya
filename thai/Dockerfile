# Training/eval image for the laya Thai fine-tune experiment.
# Reuses the OpenThai-SystemOne serving image (torch cu130 + transformers 5.x); the fork root is mounted at /work.
FROM openthai-systemone:0.1.0-batched
RUN pip install --no-cache-dir "datasets>=3.0" && rm -rf /root/.cache/pip
WORKDIR /work
ENV HF_HOME=/hf PYTHONPATH=/work:/work/thai/ots PYTHONUNBUFFERED=1
ENTRYPOINT []
CMD ["bash"]
