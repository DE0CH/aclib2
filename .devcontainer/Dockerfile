FROM archlinux:latest

RUN pacman -Sy --noconfirm archlinux-keyring
RUN pacman -Syu --noconfirm git base-devel sudo zsh thefuck
RUN echo "vscode ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

ARG UNAME=vscode
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${UNAME}
RUN useradd -m -u ${UID} -g ${GID} -s /bin/zsh ${UNAME}

RUN chsh -s /bin/zsh vscode

USER vscode 
RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/DE0CH/dotfiles/master/setup-zsh.sh)"
WORKDIR /home/vscode
RUN git clone https://aur.archlinux.org/yay.git
WORKDIR /home/vscode/yay
RUN makepkg -si --noconfirm

USER root

# Project specific stuff
RUN pacman -Syu --noconfirm zsh git python3 