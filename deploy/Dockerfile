FROM python:3.11-slim-bookworm AS gmssl-build
ARG GMSSL_COMMIT=ed20e56a016013ca663a0c202ee5f96edf4bc9ad
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN git init GmSSL && cd GmSSL && git remote add origin https://github.com/guanzhi/GmSSL.git && git fetch --depth 1 origin "$GMSSL_COMMIT" && git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "$GMSSL_COMMIT"
RUN cmake -S GmSSL -B out -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/gmssl -DBUILD_SHARED_LIBS=ON && cmake --build out -j2 && cmake --install out

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=gmssl-build /opt/gmssl/ /usr/local/
RUN ldconfig && useradd --uid 10001 --create-home facecheck
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend
COPY scripts ./scripts
RUN pip install --no-cache-dir .
ENV FACECHECK_DEPLOYMENT=cloud PYTHONPATH=/app/src PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FACECHECK_DATA_ROOT=/var/lib/facecheck FACECHECK_CRYPTO_BACKEND=auto FACECHECK_SM4_NATIVE_IMPL=gmssl FACECHECK_YUNET_MODEL=/app/models/face_detection_yunet_2023mar.onnx FACECHECK_ARCFACE_MODEL=/app/models/arcface_w600k_r50.onnx FACECHECK_SFACE_MODEL=/app/models/face_recognition_sface_2021dec.onnx FACECHECK_RECOGNITION_MODEL=arcface
RUN python -c "from facecheck.crypto_backends import get_crypto_backend; assert get_crypto_backend().name == 'gmssl-native'"
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "facecheck.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--workers", "1", "--limit-concurrency", "64"]
