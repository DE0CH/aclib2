FROM debian:bullseye
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "deb http://deb.debian.org/debian stretch main" >> /etc/apt/sources.list

RUN apt-get update
RUN apt-get install -y git sudo zsh thefuck bash curl build-essential python3 python3-pip swig libnuma-dev  openjdk-8-jdk
RUN cd /usr/bin && ln -s python3 python
RUN echo "vscode ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN useradd -m vscode -s /bin/zsh

USER vscode 
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DE0CH/dotfiles/master/setup-zsh.sh)"
WORKDIR /home/vscode

