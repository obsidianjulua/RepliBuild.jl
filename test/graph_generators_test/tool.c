#include <stdio.h>

int main(int argc, char **argv) {
    FILE *f;
    if (argc < 2 || !(f = fopen(argv[1], "w")))
        return 1;
    fputs("#define TOOL_OUT 7\n", f);
    return fclose(f) != 0;
}
