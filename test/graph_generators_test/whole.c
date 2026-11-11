/* gg_whole: its own unit plus gg_parts' objects. */
int gg_part_twice(int x);
int gg_whole_quad(int x) { return gg_part_twice(gg_part_twice(x)); }
