FROM debian:bullseye

RUN apt-get update
RUN apt-get install -y git sudo zsh thefuck bash curl htop
RUN echo "vscode ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN useradd -m vscode -s /bin/zsh 

USER vscode 
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DE0CH/dotfiles/master/setup-zsh.sh)"
RUN echo 'eval $(thefuck --alias)' >> /home/vscode/.zshrc
WORKDIR /home/vscode

USER root
# Project specific stuff
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y build-essential libboost-all-dev swig doxygen git cmake python3 python3-pip ruby-dev lcov
RUN gem install coveralls-lcov
USER vscode
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user -U pip-tools
RUN echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> /home/vscode/.zshenv
RUN pip3 install ipython
