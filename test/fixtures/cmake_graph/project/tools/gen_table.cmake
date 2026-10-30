file(WRITE "${OUT}" "int core_table(int i) { static const int t[] = {1, 2, 3}; return t[i]; }\n")
