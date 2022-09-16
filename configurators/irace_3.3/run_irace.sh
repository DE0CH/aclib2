export PATH="$(Rscript -e "cat(paste0(system.file(package='irace', 'bin', mustWork=TRUE), ':'))" 2> /dev/null)${PATH}"
IRACE=$(which irace)
if [ ! -x $IRACE ]; then
    echo "$0: error: $IRACE not found or not executable"
    exit 1
fi

BINDIR="$(dirname "$(readlink -f "$IRACE")")"
export R_LIBS=${BINDIR%irace/bin}:"$R_LIBS"

$IRACE -s $1 --debug-level 0 --parallel 1 --capping 1 --bound-par 10  

