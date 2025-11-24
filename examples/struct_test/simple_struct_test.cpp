struct Point {
    double x;
    double y;
};

Point create_point(double x, double y) {
    return {x, y};
}

Point add_points(Point a, Point b) {
    return {a.x + b.x, a.y + b.y};
}

double distance(Point a, Point b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx*dx + dy*dy;
}
