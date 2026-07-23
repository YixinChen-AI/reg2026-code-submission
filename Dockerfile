FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ARG APP_VERSION=0.6.0

LABEL org.opencontainers.image.title="CYX-AI REG2026"
LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.authors="Yixin Chen"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN groupadd --system user \
    && useradd --system --create-home --gid user user

WORKDIR /opt/app

COPY requirements.txt ./
RUN python -m pip install \
      --no-cache-dir \
      --disable-pip-version-check \
      --no-color \
      --requirement requirements.txt

COPY --chown=user:user model/ /opt/app/model/
COPY configs/artifacts-v0.6.0.json /opt/app/configs/artifacts-v0.6.0.json
COPY scripts/verify_model_assets.py /opt/app/scripts/verify_model_assets.py

RUN python /opt/app/scripts/verify_model_assets.py \
      --root /opt/app/model \
      --lock /opt/app/configs/artifacts-v0.6.0.json \
    && chown -R user:user /opt/app/model

COPY --chown=user:user core.py inference.py /opt/app/
COPY --chown=user:user src/contracts.py /opt/app/src/contracts.py
COPY --chown=user:user src/interf0/ /opt/app/src/interf0/
COPY --chown=user:user src/interf1/__init__.py src/interf1/model.py /opt/app/src/interf1/

RUN ln -s /opt/app/model/exemplar_bank.npz /opt/app/src/interf1/exemplar_bank.npz \
    && ln -s /opt/app/model/exemplar_cots.json /opt/app/src/interf1/exemplar_cots.json

USER user

ENTRYPOINT ["python", "inference.py"]
