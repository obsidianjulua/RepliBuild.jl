# Reads GG_GEN_VALUE out of the generated header IN and writes OUT from it.
file(READ "${IN}" content)
string(REGEX MATCH "GG_GEN_VALUE ([0-9]+)" _match "${content}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "no GG_GEN_VALUE in ${IN}")
endif()
file(WRITE "${OUT}" "#define GG_TABLE_VALUE (${CMAKE_MATCH_1} * 2)\n")
