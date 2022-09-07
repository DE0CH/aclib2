FROM alpine:latest

RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/testing/" >> /etc/apk/repositories
RUN apk update
RUN apk add git sudo zsh thefuck bash curl
RUN echo "vscode ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN adduser -D -s /bin/zsh vscode 

USER vscode 
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DE0CH/dotfiles/master/setup-zsh.sh)"
WORKDIR /home/vscode

USER root
# Project specific stuff
RUN apk add python3 py3-pip 
