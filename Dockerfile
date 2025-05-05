# Base image
FROM python:3.11
#FROM ubuntu:latest

WORKDIR /home/trending
LABEL maintainer="Stefan Bogdanel <stefan@izdrail.com>"

# Install dependencies
RUN apt update && apt install -y \
    curl \
    nodejs \
    npm \
    mlocate \
    net-tools \
    software-properties-common \
    openjdk-17-jdk \
    maven \
    chromium \
    && apt-get clean

# Install pip packages and supervisord
RUN pip install --no-cache-dir --upgrade pip \
    && pip install supervisor pipx 




WORKDIR /home/skraper
RUN git clone https://github.com/sokomishalov/skraper.git .
RUN pwd
WORKDIR /home/skraper
RUN ./mvnw clean package -DskipTests=true \
    && mkdir -p /usr/local/skraper \
    && cp /home/skraper/cli/target/cli.jar /usr/local/skraper/
RUN echo '#!/bin/bash\njava -jar /usr/local/skraper/cli.jar "$@"' > /usr/local/bin/skraper \
    && chmod +x /usr/local/bin/skraper

WORKDIR /home/trending
# Install Python packages
COPY ./requirements.txt /home/trending/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /home/trending/requirements.txt \
    && pip install trendspy justext text2emotion pymupdf4llm python-multipart sqlalchemy yake fastapi_versioning tls_client uvicorn gnews \
    && python3 -m nltk.downloader -d /usr/local/share/nltk_data wordnet punkt stopwords vader_lexicon \
    && python3 -m textblob.download_corpora


# Customize shell with Zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

COPY . .


# Supervisord configuration
COPY docker/supervisord.conf /etc/supervisord.conf

# Update database
RUN updatedb


# Expose application port
EXPOSE 1099



# Run application
ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf", "-n"]