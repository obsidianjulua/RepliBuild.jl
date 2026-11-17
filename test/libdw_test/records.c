typedef struct { char c; short s[3]; } __attribute__((aligned(16))) P;
struct node { struct node *next; _Bool b; _Complex double z; __int128 w; };
P pfn(P p, struct node *n) { return p; }
int vfn(int n, ...) { return n; }
typedef float v4 __attribute__((vector_size(16)));
v4 vadd(v4 a, v4 b) { return a + b; }
