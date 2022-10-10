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

USER root
RUN apt-get install -y unzip r-base
RUN Rscript -e "install.packages('irace', repos='https://cloud.r-project.org')"

USER vscode
RUN echo 'export PATH=/usr/local/lib/R/site-library/irace/bin:$PATH' >> /home/vscode/.bashrc
RUN echo 'export PATH=/usr/local/lib/R/site-library/irace/bin:$PATH' >> /home/vscode/.zshenv

USER root
RUN apt-get install -y build-essential libboost-all-dev swig doxygen git cmake python3 python3-pip ruby-dev lcov
RUN gem install coveralls-lcov
USER vscode
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user -U pip-tools
RUN echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> /home/vscode/.zshenv
RUN pip3 install ipython
